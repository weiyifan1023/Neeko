import json
import os
from typing import Optional, List, Tuple, Dict

import peft
import moelora
import torch
from transformers.trainer import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, TRAINER_STATE_NAME
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoConfig, BitsAndBytesConfig, \
    AutoModelForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.utils.versions import require_version

from .utils import get_logger, ModelArguments, FinetuningArguments
from .utils import VALUE_HEAD_FILE_NAME


logger = get_logger(__name__)


def prepare_model_for_training(
        model: PreTrainedModel,
        finetuning_type: str,
        output_embedding_layer_name: Optional[str] = "lm_head",
        use_gradient_checkpointing: Optional[bool] = True,
        layer_norm_names: Optional[List[str]] = ["norm", "ln_f"] # for LLaMA and BLOOM setting
) -> PreTrainedModel:

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if finetuning_type != "full" and hasattr(model, output_embedding_layer_name):
        output_embedding_layer: torch.nn.Linear = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
                trainable_params, all_param, 100 * trainable_params / all_param))


def get_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]: # get state dict containing trainable parameters
    state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in model.named_parameters():
        if v.requires_grad:
            filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

    return filtered_state_dict


def load_trainable_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    weights_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if os.path.exists(weights_file):
        model_state_dict = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False) # skip missing keys
    elif os.path.exists(os.path.join(checkpoint_dir, WEIGHTS_INDEX_NAME)):
        load_sharded_checkpoint(model, checkpoint_dir, strict=False)
    else:
        logger.warning("Provided path ({}) does not contain pre-trained weights.".format(checkpoint_dir))
        return False
    return True


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
    if not os.path.exists(valuehead_file):
        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
    model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))
    return True


def smooth(scalars: List[float], weight: Optional[float] = 0.9) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def _init_adapter(
        model: PreTrainedModel,
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: bool,
        is_mergeable: bool
) -> PreTrainedModel:
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full":
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze":
        logger.info("Fine-tuning method: Freeze")

        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in finetuning_args.trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

        if model_args.checkpoint_dir is not None:
            assert load_trainable_params(model, model_args.checkpoint_dir[0]), "Model checkpoint is not correctly loaded."
    
    if finetuning_args.finetuning_type == "lora":
        from peft import PeftModel, LoraConfig, TaskType, get_peft_model
        logger.info("Fine-tuning method: LoRA")
        lastest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], peft.utils.WEIGHTS_NAME)), \
                "Provided path ({}) does not contain a LoRA weight.".format(model_args.checkpoint_dir[0])
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], peft.utils.CONFIG_NAME)), \
                "The given checkpoint may be not a LoRA checkpoint, please specify `--finetuning_type full/freeze` instead."

            if (is_trainable and model_args.resume_lora_training) or (not is_mergeable): # continually train on the lora weights
                checkpoints_to_merge, lastest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if lastest_checkpoint is not None: # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=is_trainable)

        if is_trainable and lastest_checkpoint is None: # create new lora weights while training
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, lora_config)
        
    if finetuning_args.finetuning_type == "moelora":
        from moelora import PeftModel, MoeLoraConfig, TaskType, get_peft_model
        logger.info("Fine-tuning method: MoeLoRA")
        lastest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], moelora.utils.WEIGHTS_NAME)), \
                "Provided path ({}) does not contain a LoRA weight.".format(model_args.checkpoint_dir[0])
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], moelora.utils.CONFIG_NAME)), \
                "The given checkpoint may be not a LoRA checkpoint, please specify `--finetuning_type full/freeze` instead."

            if (is_trainable and model_args.resume_lora_training) or (not is_mergeable): # continually train on the lora weights
                checkpoints_to_merge, lastest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if lastest_checkpoint is not None: # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=is_trainable)

        if is_trainable and lastest_checkpoint is None: # create new lora weights while training
            moe_lora_config = MoeLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                num_moe=finetuning_args.num_moe,
                gating=finetuning_args.gating,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, moe_lora_config)

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model


def load_pretrained(
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: Optional[bool] = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
        **config_kwargs
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000: # 64000 for baichuan model (older version)
        tokenizer.pad_token_id = 0 # set as the <unk> token

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    is_mergeable = True

    # Quantization configurations (using bitsandbytes library).
    if model_args.quantization_bit is not None:
        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("transformers>=4.30.1", "To fix: pip install transformers>=4.30.1")
            require_version("accelerate>=0.20.3", "To fix: pip install accelerate>=0.20.3")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )
        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    if not is_trainable: # `device_map=auto` should be used for inference only
        config_kwargs["device_map"] = "auto"

    if model_args.checkpoint_dir is not None and finetuning_args.finetuning_type == "full":
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=torch.bfloat16 if model_args.compute_dtype == torch.bfloat16 else torch.float16,
        low_cpu_mem_usage=True,
        **config_kwargs
    )
    model = prepare_model_for_training(model, finetuning_args.finetuning_type) if is_trainable else model
    model = _init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)

    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.half() if model_args.quantization_bit is None else model # cast from fp32 to fp16

    print_trainable_params(model)

    return model, tokenizer


def plot_loss(save_dictionary: os.PathLike, keys: Optional[List[str]] = ["loss"]) -> None:
    import matplotlib.pyplot as plt
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(save_dictionary, "training_{}.png".format(key)), format="png", dpi=100)
        print("Figure saved:", os.path.join(save_dictionary, "training_{}.png".format(key)))

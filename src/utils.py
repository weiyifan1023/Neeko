import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict, Any, Tuple

import datasets
import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, LogitsProcessorList, LogitsProcessor

IGNORE_INDEX = -100
VALUE_HEAD_FILE_NAME = "value_head.bin"
FINETUNING_ARGS_NAME = "finetuning_args.json"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co."}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Will use the token generated when running `huggingface-cli login`."}
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."}
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use double quantization in int4 training or not."}
    )
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        metadata={"help": "Used in quantization configs. Do not specify this argument manually."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory(s) containing the delta model checkpoints as well as the configurations."}
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to plot the training loss after fine-tuning or not."}
    )

    def __post_init__(self):
        if self.checkpoint_dir is not None:  # support merging multiple lora weights
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]

        if self.quantization_bit is not None:
            assert self.quantization_bit in [4, 8], "We only accept 4-bit or 8-bit quantization."


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: str = field(
        default="character-llm",
        metadata={"help": "Dataset name"}
    )
    data_path: str = field(
        default="/path/to/datasets/character-llm-data/prompted/shuffle.jsonl",
        metadata={"help": "Dataset Path"}
    )
    embds_dir: str = field(
        default="/path/to/role_embds",
        metadata={"help": "Role Embeddings dir"}
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total output sequence length after tokenization."}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."}
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"}
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    dev_ratio: Optional[float] = field(
        default=0,
        metadata={"help": "Proportion of the dataset to include in the development set, should be between 0.0 and 1.0."}
    )


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[Literal["none", "freeze", "moelora", "lora", "full"]] = field(
        default="moelora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    num_hidden_layers: Optional[int] = field(
        default=32,
        metadata={"help": "Number of decoder blocks in the model. \
                  LLaMA choices: [\"32\", \"40\", \"60\", \"80\"], \
                  BLOOM choices: [\"24\", \"30\", \"70\"], \
                  Baichuan choices: [\"32\"]"}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for Freeze fine-tuning."}
    )
    name_module_trainable: Optional[Literal["mlp", "self_attn", "self_attention"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for Freeze fine-tuning. \
                  LLaMA choices: [\"mlp\", \"self_attn\"], \
                  BLOOM choices: [\"mlp\", \"self_attention\"], \
                  Baichuan choices: [\"mlp\", \"self_attn\"]"}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    num_moe: Optional[int] = field(
        default=8,
        metadata={"help": "The num of experts"}
    )
    gating: Optional[str] = field(
        default="Standard",
        metadata={"help": "Select the Gating Network"}
    )
    lora_alpha: Optional[int] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning (similar with the learning rate)."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )
    lora_target: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Name(s) of target modules to apply LoRA. Use comma to separate multiple modules. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  BLOOM choices: [\"query_key_value\", \"self_attention.dense\", \"mlp.dense\"], \
                  Baichuan choices: [\"W_pack\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"]"}
    )

    def __post_init__(self):
        if isinstance(self.lora_target, str): # support custom target modules/layers of LoRA
            self.lora_target = [target.strip() for target in self.lora_target.split(",")]

        if self.num_layer_trainable > 0: # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [self.num_hidden_layers - k for k in range(self.num_layer_trainable)]
        else: # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-self.num_layer_trainable)]

        self.trainable_layers = ["{:d}.{}".format(idx, self.name_module_trainable) for idx in trainable_layer_ids]

        assert self.finetuning_type in ["none", "freeze", "moelora", "lora", "full"], "Invalid fine-tuning method."

    def save_to_json(self, json_path: str):
        """Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


@dataclass
class GeneratingArguments:
    """
    Arguments pertaining to specify the decoding parameters.
    """
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."}
    )
    temperature: Optional[float] = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_p: Optional[float] = field(
        default=0.7,
        metadata={"help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."}
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."}
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."}
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."}
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", None):
            args.pop("max_length", None)
        return args


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


logger = get_logger(__name__)


def prepare_args() -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # Provide arguments with a json file.
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    assert not (training_args.do_train and training_args.predict_with_generate), \
        "`predict_with_generate` cannot be set as True while training."

    assert (not training_args.do_predict) or training_args.predict_with_generate, \
        "Please enable `predict_with_generate` to save model predictions."

    assert model_args.quantization_bit is None or finetuning_args.finetuning_type == "lora", \
        "Quantization is only compatible with the LoRA method."

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            assert len(model_args.checkpoint_dir) == 1, "Only LoRA tuning accepts multiple checkpoints."
        else:
            assert model_args.quantization_bit is None or len(model_args.checkpoint_dir) == 1, \
                "Quantized model only accepts a single checkpoint."

    if model_args.quantization_bit is not None and (not training_args.do_train):
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if training_args.do_train and (not training_args.fp16):
        logger.warning("We recommend enable fp16 mixed precision training.")

    if training_args.local_rank != -1 and training_args.ddp_find_unused_parameters is None:
        logger.warning("`ddp_find_unused_parameters` needs to be set as False in DDP training.")
#         training_args.ddp_find_unused_parameters = False

#     training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim # suppress warning

    if model_args.quantization_bit is not None:
        if training_args.fp16:
            model_args.compute_dtype = torch.float16
        elif training_args.bf16:
            model_args.compute_dtype = torch.bfloat16
        else:
            model_args.compute_dtype = torch.float32

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"  distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args


def prepare_infer_args() -> Tuple[ModelArguments, DataTrainingArguments, FinetuningArguments, GeneratingArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuningArguments, GeneratingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # Provide arguments with a json file.
        model_args, data_args, finetuning_args, generating_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, finetuning_args, generating_args = parser.parse_args_into_dataclasses()

    assert model_args.quantization_bit is None or finetuning_args.finetuning_type == "lora", \
        "Quantization is only compatible with the LoRA method."

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            assert len(model_args.checkpoint_dir) == 1, "Only LoRA tuning accepts multiple checkpoints."
        else:
            assert model_args.quantization_bit is None or len(model_args.checkpoint_dir) == 1, \
                "Quantized model only accepts a single checkpoint."

    return model_args, data_args, finetuning_args, generating_args


class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 0] = 1.0
        return scores


def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    return logits_processor

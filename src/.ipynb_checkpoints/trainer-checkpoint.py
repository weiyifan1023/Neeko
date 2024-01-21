from typing import List

from datasets import concatenate_datasets

from .data_collator import DynamicDataCollatorWithPadding
from .dataset import (
    LLaMaDataset, preprocess_data,

)
from .model import load_pretrained, plot_loss
from .peft_trainer import LogCallback
from .seq2seq import Seq2SeqPeftTrainer, ComputeMetrics
from .template import Template
from .utils import prepare_args, get_logits_processor


class LLaMaTrainer:
    def __init__(self, dataset_list: List[LLaMaDataset], template: Template):
        self.model_args, self.data_args, self.training_args, self.finetuning_args = prepare_args()
        if len(dataset_list) == 1:
            self.dataset = dataset_list[0].dataset
        else:
            self.dataset = concatenate_datasets([x.dataset for x in dataset_list])
        self.model, self.tokenizer = load_pretrained(self.model_args, self.finetuning_args, self.training_args.do_train)
        self.dataset = preprocess_data(template, self.dataset, self.tokenizer, self.data_args, self.training_args)
        self.data_collator = DynamicDataCollatorWithPadding(
            tokenizer=self.tokenizer,
            ignore_pad_token_for_loss=(self.data_args.ignore_pad_token_for_loss and not self.training_args.predict_with_generate)
        )

        # Override the decoding parameters of Seq2SeqTrainer
#         self.training_args.generation_max_length = self.training_args.generation_max_length if \
#             self.training_args.generation_max_length is not None else self.data_args.max_target_length
#         self.training_args.generation_num_beams = self.data_args.eval_num_beams if \
#             self.data_args.eval_num_beams is not None else self.training_args.generation_num_beams

        # Split the dataset
        if self.training_args.do_train:
            if self.data_args.dev_ratio > 1e-6:
                self.dataset, _ = self.dataset.train_test_split(test_size=self.data_args.dev_ratio)
                trainer_kwargs = {"train_dataset": self.dataset["train"], "eval_dataset": self.dataset["test"]}
            else:
                trainer_kwargs = {"train_dataset": self.dataset}
        else:  # do_eval or do_predict
            trainer_kwargs = {"eval_dataset": self.dataset}

        # Initialize our Trainer
        self.trainer = Seq2SeqPeftTrainer(
            finetuning_args=self.finetuning_args,
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[LogCallback()],
            compute_metrics=ComputeMetrics(self.tokenizer) if self.training_args.predict_with_generate else None,
            **trainer_kwargs
        )

        # Keyword arguments for `model.generate`
        self.gen_kwargs = {
            "do_sample": True,
            "top_p": 0.7,
            "max_new_tokens": self.data_args.max_target_length + 1,
            "temperature": 0.95,
            "logits_processor": get_logits_processor()
        }

    def train(self):
        # Training
        if self.training_args.do_train:
            train_result = self.trainer.train()
            self.trainer.log_metrics("train", train_result.metrics)
            self.trainer.save_metrics("train", train_result.metrics)
            self.trainer.save_state()
            self.trainer.save_model()
            if self.trainer.is_world_process_zero() and self.model_args.plot_loss:
                plot_loss(self.training_args.output_dir, keys=["loss", "eval_loss"])

    def eval(self):
        # Evaluation
        if self.training_args.do_eval:
            metrics = self.trainer.evaluate(metric_key_prefix="eval", **self.gen_kwargs)
            if self.training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
                metrics.pop("eval_loss", None)
            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)

    def test(self):
        # Predict
        if self.training_args.do_predict:
            predict_results = self.trainer.predict(self.dataset, metric_key_prefix="predict", **self.gen_kwargs)
            if self.training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
                predict_results.metrics.pop("predict_loss", None)
            self.trainer.log_metrics("predict", predict_results.metrics)
            self.trainer.save_metrics("predict", predict_results.metrics)
            self.trainer.save_predictions(predict_results)

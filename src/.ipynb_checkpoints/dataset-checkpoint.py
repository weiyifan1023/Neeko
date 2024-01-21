import json
from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments

from .template import Template
from .utils import DataTrainingArguments, IGNORE_INDEX
import pandas as pd
import csv


def preprocess_data(
        prompt_template: Template,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_args: DataTrainingArguments,
        training_args: Seq2SeqTrainingArguments
) -> Dataset:
    column_names = list(dataset.column_names)

    # support question with a single answer or multiple answers
    def get_dialog(examples):
        for i in range(len(examples["prompt"])):
            if examples["prompt"][i] and examples["response"][i]:
                query, answer = examples["prompt"][i], examples["response"][i]
                query = query + "\n" + examples["query"][i] if examples["query"][i] else query
                prefix = examples["prefix"][i] if examples["prefix"][i] else ""
                dialog = prompt_template.get_dialog(query, answer, examples["history"][i], prefix)
                yield dialog

    def preprocess_supervised_dataset(examples):
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for input with history, we build multiple input-label pairs just like:
        # https://github.com/lm-sys/FastChat/blob/f17c092f64840fa6354ed52789dccb2daa793d0b/fastchat/train/train.py#L112
        model_inputs = {"input_ids": [], "labels": []}
        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=False)
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                if len(source_ids) > data_args.max_source_length - 1:  # bos token
                    source_ids = source_ids[:data_args.max_source_length - 1]
                if len(target_ids) > data_args.max_target_length - 1:  # eos token
                    target_ids = target_ids[:data_args.max_target_length - 1]

                input_ids += [tokenizer.bos_token_id] + source_ids + target_ids + [tokenizer.eos_token_id]
                labels += [IGNORE_INDEX] * (len(source_ids) + 1) + target_ids + [tokenizer.eos_token_id]

            if len(input_ids) > data_args.max_source_length + data_args.max_target_length:
                input_ids = input_ids[:data_args.max_source_length + data_args.max_target_length]
            if len(labels) > data_args.max_source_length + data_args.max_target_length:
                labels = labels[:data_args.max_source_length + data_args.max_target_length]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))

    preprocess_function = preprocess_supervised_dataset

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )
        print_supervised_dataset_example(dataset[0])

        return dataset


class LLaMaDataset(ABC):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataset = self.__read_data_to_huggingface_dataset__(data_path)

    @abstractmethod
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        """
        Reading the data and preprocessing to the huggingface dataset with the column_names bellow:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        :return: dataset: the huggingface Dataset
        """
        pass


class MultiHiertt(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
            query = ""
            for text_evidence in one["qa"]["text_evidence"]:
                query += (one["paragraphs"][text_evidence] + "\n")

            for table_evidence in one["qa"]["table_evidence"]:
                query += (one["table_description"][table_evidence] + "\n")
            dataset.append({
                "prefix": None,
                "prompt": "According to the information, use the operation in "
                          "[add, subtract, multiply, divide, exp, greater, table_sum, table_average, table_max, table_min] "
                          "to construct a program to answer the question:" + one["qa"]["question"],
                "query": query,
                "response": one["qa"]["program"],
                "history": None
            })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class SKG(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
            dataset.append({
                "prefix": None,
                "prompt": "According to the table, try your best to answer the question: " + one["text_in"],
                "query": one["struct_in"],
                "response": one["seq_out"],
                "history": None
            })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class CompAQT(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
            texts = "paragraphs: \n"
            tables = "table descriptions: \n"
            for k, v in one["qa"]["gold_inds"].items():
                if "text" in k:
                    texts += (v + "\n")
                if "table" in k:
                    tables += (v + "\n")
            dataset.append({
                "prefix": None,
                "prompt": "According to the paragraphs and table descriptions, try your best to answer the question: " + one["qa"]["question"],
                "query": texts + tables,
                "response": one["qa"]["program"],
                "history": None
            })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class MNLIM(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts = f"sentence1: {data['sentence1']}\nsentence2: {data['sentence2']}\nanswer: "
                dataset.append({
                    "prefix": None,
                    "prompt": "Give you two sentences, try your best to identify the relationship between them. Your answer must be neutral, entailment or contradiction.",
                    "query": texts,
                    "response": data['annotator_labels'],
                    "history": None
                })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class SQUAD(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for part in data['data']:
            part_data = part['paragraphs']
            for one_data in part_data:
                paragraph = one_data['context']
                for qa in one_data['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]['text'] if qa['is_impossible'] == False else ' ' # there are many possible answers, default to use answer[0]
                    texts = f'paragraph:{paragraph}\nquestion: {question}\n'
                    dataset.append({
                        "prefix": None,
                        "prompt": "According to the given paragraph, try your best to answer a question by selecting a span from the paragraph or give a blank line when the question is unanswerable.",
                        "query": texts,
                        "response": answer,
                        "history": None
                    })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class HotPotQA(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str, test: bool = False) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
            paragraph = 'paragraphs:\n'
            question = one['question']
            if test:
                answer = ''
            else:
                answer = one['answer']
            for context in one['context']:
                one_context = f'{context[0]}: \n'
                for context_item in context[1]:
                    one_context = one_context + context_item
                paragraph = paragraph + '\n' + one_context
            
            dataset.append({
                "prefix": None,
                "prompt": "According to the paragraphs, try your best to answer the question: " + question,
                "query": paragraph,
                "response": answer,
                "history": None
            })
                

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset
    

class AdversarialQA(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str, test: bool = False) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for part in data['data']:
            part_data = part['paragraphs']
            for one_data in part_data:
                paragraph = one_data['context']
                for qa in one_data['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]['text']
                    texts = f'paragraph:{paragraph}\nquestion: {question}\n'
                    dataset.append({
                        "prefix": None,
                        "prompt": "According to the given paragraph, try your best to answer a question by selecting a span from the paragraph.",
                        "query": texts,
                        "response": answer,
                        "history": None
                    })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class CoLA(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str, test: bool = False) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = csv.reader(
            open(data_path, 'r', encoding='utf-8'), delimiter='\t'
        )
        dataset = []
        for one in data:
            seq = one[3]
            label = one[1]
            dataset.append({
                "prefix": None,
                "prompt": "Here is an English sentence for you to determine if the sentence grammar is correct. Output 1 correctly and 0 uncorrectly.",
                "query": "sentence:\n" + seq,
                "response": label,
                "history": None
            })
        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class RTE(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str, test: bool = False) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = csv.reader(
            open(data_path, 'r', encoding='utf-8'), delimiter='\t'
        )
        dataset = []
        for one in data:
            if one[0] == "index" or len(one) < 4:
                continue
            seq1 = one[1]
            seq2 = one[2]
            label = one[3]
            text = "sentence1:\n" + seq1 + "\nsentence2:\n" + seq2
            dataset.append({
                "prefix": None,
                "prompt": "Recognize the textual entailment between sentence1 and sentence2, Output entailment or not_entailment.",
                "query": text,
                "response": label,
                "history": None
            })
        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class SST2(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str, test: bool = False) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = csv.reader(
            open(data_path, 'r', encoding='utf-8'), delimiter='\t'
        )
        dataset = []
        for one in data:
            if one[0] == "sentence" or len(one) < 2:
                continue
            seq = one[0]
            dataset.append({
                "prefix": None,
                "prompt": "Recognize the emotions in the given movie review is positive or negative. Output 1 positive and 0 negative.",
                "query": "movie review:\n" + seq,
                "response": one[1],
                "history": None
            })
        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class QNLI(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str, test: bool = False) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = csv.reader(
            open(data_path, 'r', encoding='utf-8'), delimiter='\t'
        )
        dataset = []
        for one in data:
            if one[0] == "index" or len(one) < 4:
                continue
            question = one[1]
            seq = one[2]
            label = one[3]
            text = "question:\n" + question + "\nsentence:\n" + seq
            dataset.append({
                "prefix": None,
                "prompt": "Recognize the textual entailment between question and sentence, Output entailment or not_entailment.",
                "query": text,
                "response": label,
                "history": None
            })
        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class MRPC(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str, test: bool = False) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = csv.reader(
            open(data_path, 'r', encoding='utf-8'), delimiter='\t'
        )
        dataset = []
        for one in data:
            if "Quality" in one[0] or len(one) < 5:
                continue
            quality = one[0]
            string_1 = one[3]
            string_2 = one[4]
            text = "sentence1:\n" + string_1 + "\nsentence2:\n" + string_2
            dataset.append({
                "prefix": None,
                "prompt": "Recognize the the semantic similarity between sentence1 and sentence2, Output 1 Similar and 0 dissimilar.",
                "query": text,
                "response": quality,
                "history": None
            })
        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class STSB(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str, test: bool = False) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = csv.reader(
            open(data_path, 'r', encoding='utf-8'), delimiter='\t'
        )
        dataset = []
        for one in data:
            if one[0] == "index" or len(one) < 10:
                continue
            seq1 = one[-3]
            seq2 = one[-2]
            score = one[-1]
            text = "sentence1:\n" + seq1 + "\nsentence2:\n" + seq2
            dataset.append({
                "prefix": None,
                "prompt": "Rate the semantic similarity between sentence1 and sentence2 on a scale of 1.000 to 5.000.",
                "query": text,
                "response": score,
                "history": None
            })
        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class LLMBenchmark(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
            dataset.append({
                "prefix": None,
                "prompt": one["instruction"],
                "query": one["input"],
                "response": one["answer"],
                "history": None
            })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset

class comp(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
            dataset.append({
                "prefix": None,
                "prompt": one["instruction"],
                "query": one["input"],
                "response": one["output"],
                "history": None
            })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset

STR_DATASET_MAP = {
    "multi_hiertt": MultiHiertt,
    "skg": SKG,
    "wikitq": SKG,
    "hybridqa": SKG,
    "compaqt": CompAQT,
    "mnlim": MNLIM,
    "squad": SQUAD,
    "hotpotqa": HotPotQA,
    "adversarialqa": AdversarialQA,
    "sst2": SST2,
    "qnli": QNLI,
    "mrpc": MRPC,
    "stsb": STSB,
    "cola": CoLA,
    "rte": RTE,
    "addsub": LLMBenchmark,
    "aqua": LLMBenchmark,
    "arcc": LLMBenchmark,
    "arce": LLMBenchmark,
    "boolq": LLMBenchmark,
    "gsm8k": LLMBenchmark,
    "obqa": LLMBenchmark,
    "piqa": LLMBenchmark,
    "multiarith": LLMBenchmark,
    "singleeq": LLMBenchmark,
    "svamp": LLMBenchmark,
    "comp": comp,
    "math": comp,
    "common": LLMBenchmark,
}
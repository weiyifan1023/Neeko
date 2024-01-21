# TextTableQAKit/llama

## QuickStart

### MultiHiertt Dataset as a demonstration
```
from llama import LLaMaTrainer
from llama import MultiHiertt
from llama import defaultTemplate

template = defaultTemplate()
datasets = MultiHiertt("path of your dataset")
trainer = Trainer([datasets], defaultTemplate)
trainer.train()
```

### start train
```
python main.py \
--model_name_or_path ./ckpt/llama-7b-hf \
--do_train \
--finetuning_type lora \
--output_dir ./ckpt/lora \
--max_source_length 1536 \
--overwrite_cache \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 50 \
--save_steps 1000 \
--learning_rate 5e-5 \
--num_train_epochs 2 \
--plot_loss \
--lora_rank 32 \
--fp16
```


### Add new datasets
```
import json
from datasets import Dataset
from llama import LLaMaTrainer
from llama import LLaMaDataset
from llama import Template


class NewDataset(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(your_data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
			# process your data to:
            dataset.append({
                "prefix": prefix,
                "prompt": prompt,
                "query": query,
                "response": response,
                "history": history
            })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class NewTemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            prompt="Human: {query}\nAssistant: ",
            sep="\n",
            use_history=False
        )


newtemplate = NewTemplate()
datasets = NewDataset()
trainer = Trainer([datasets], newtemplate)
trainer.train()
```

## Training with multi-datasets
```
dataset1 = Dataset1("path1")
dataset2 = Dataset1("path2")
...

trainer = Trainer([dataset1, dataset2, ...], template)
trainer.train()
```
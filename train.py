from src import STR_TEMPLATE_MAP, LLaMaTrainer, STR_DATASET_MAP, prepare_args


_, data_args, _, _ = prepare_args()
dataset_name = data_args.dataset
data_dir = "/path/to/datasets/"
dataCLS = STR_DATASET_MAP[dataset_name]
template = STR_TEMPLATE_MAP[dataset_name]()

if dataset_name == "character-llm":
    datasets = dataCLS(data_dir + "character-llm-data/prompted/shuffle.jsonl")
elif dataset_name == "character-llm-single":
    datasets = dataCLS(data_args.data_path)
else:
    raise NotImplementedError
trainer = LLaMaTrainer([datasets], template)
trainer.train()

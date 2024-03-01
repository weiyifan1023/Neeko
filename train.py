from src import STR_TEMPLATE_MAP, LLaMaTrainer, STR_DATASET_MAP, prepare_args

_, data_args, _, _ = prepare_args()
dataset_name = data_args.dataset
data_path = data_args.data_path
dataCLS = STR_DATASET_MAP[dataset_name]
template = STR_TEMPLATE_MAP[dataset_name]()

datasets = dataCLS(data_args.data_path)
trainer = LLaMaTrainer([datasets], template)
trainer.train()

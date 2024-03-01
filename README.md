<div align="center">
    <img src="https://github.com/weiyifan1023/Neeko/blob/main/images/neeko_poster.png" width="448" height="256">
</div>



# Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent

<p align="center">
<a href="https://github.com/weiyifan1023/Neeko/blob/main/LICENSE">
<img src='https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg'></a>
<img src='https://img.shields.io/badge/python-3.9+-blue.svg'>
<img src='https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg'>
</p>

<p align="center">
🔔 <a href="https://github.com/weiyifan1023/Neeko" target="_blank">Code</a> • 📃 <a href="https://arxiv.org/abs/2402.13717" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/datasets/fnlp/character-llm-data" target="_blank">Dataset</a> <br>
</p>

## Abstract
Large Language Models (LLMs) have revolutionized open-domain dialogue agents but encounter challenges in multi-character role-playing (MCRP) scenarios.
To address the issue, we present **Neeko**, an innovative framework designed for efficient multiple characters imitation.
Unlike existing methods, Neeko employs a dynamic low-rank adapter (LoRA) strategy, enabling it to adapt seamlessly to diverse characters.
Our framework breaks down the role-playing process into
agent pre-training, multiple characters playing, and character incremental learning, effectively handling both seen and unseen roles.
This dynamic approach, coupled with distinct LoRA blocks for each character, enhances Neeko's adaptability to unique attributes, personalities, and speaking patterns.
As a result, Neeko demonstrates superior performance in MCRP over most existing methods, offering more engaging and versatile user interaction experiences.

## Framework
![Image text](https://github.com/weiyifan1023/Neeko/blob/main/OverallFrame.png)

## Getting Started
```
git clone https://github.com/weiyifan1023/Neeko.git
cd Neeko
```

### 1. Shuffle data from multiple roles.
```
python shuffle_data.py \
    --data_dir /path/to/your/character-llm-data/ \
    --out_path /path/to/your/character-llm-data/prompted/shuffle.jsonl
```

### 2. Use a pretrained transformer encoder model to generate role embeddings
Here, we take [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large) as an example. 
```
python embd_roles.py \
    --encoder_path /path/to/your/deberta-v3-large \
    --seed_data_path /path/to/your/seed_data \
    --save_path /path/to/save/your/role_embds
```

### 3. Training Neeko
We take [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) as an example, replace some paths in neeko.sh, and then execute:
```
bash neeko.sh
```

## Demonstration
![Image text](https://github.com/weiyifan1023/Neeko/blob/main/images/llama-chat_dialogue.jpg)

## Citation
If you find our paper inspiring and have utilized it in your work, please cite our paper.
```
@article{yu2024neeko,
  title={Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent},
  author={Yu, Xiaoyan and Luo, Tongxu and Wei, Yifan and Lei, Fangyu and Huang, Yiming and Hao, Peng and Zhu, Liehuang},
  journal={arXiv preprint arXiv:2402.13717},
  year={2024}
}
```

## Contact
xiaoyan.yu@bit.edu.cn &&  2748113810@qq.com (Tongxu Luo) &&  weiyifan2021@ia.ac.cn (Preferred)

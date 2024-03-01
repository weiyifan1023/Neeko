# Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent

<p align="center">
<a href="https://github.com/choosewhatulike/character-llm/blob/main/LICENSE">
<img src='https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg'></a>
<img src='https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg'>
<img src='https://img.shields.io/badge/python-3.9+-blue.svg'>
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
xiaoyan.yu@bit.edu.cn  2748113810@qq.com (Tongxu Luo)  weiyifan2021@ia.ac.cn

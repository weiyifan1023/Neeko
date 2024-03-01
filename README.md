# Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent
## Abstract
Large Language Models (LLMs) have revolutionized open-domain dialogue agents but encounter challenges in multi-character role-playing (MCRP) scenarios.
To address the issue, we present **Neeko**, an innovative framework designed for efficient multiple characters imitation.
Unlike existing methods, Neeko employs a dynamic low-rank adapter (LoRA) strategy, enabling it to adapt seamlessly to diverse characters.
Our framework breaks down the role-playing process into
agent pre-training, multiple characters playing, and character incremental learning, effectively handling both seen and unseen roles.
This dynamic approach, coupled with distinct LoRA blocks for each character, enhances Neeko's adaptability to unique attributes, personalities, and speaking patterns.
As a result, Neeko demonstrates superior performance in MCRP over most existing methods, offering more engaging and versatile user interaction experiences.
## Framework
![图片alt](图片链接 "图片title")

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


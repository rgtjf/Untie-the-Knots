# Untie-the-Knots

Untie-the-Knots: An Efficient Data Augmentation Strategy for Long-Context Pre-Training in Language Models

<div align="center">
    <a href="https://huggingface.co/collections/rgtjf/utk-66daf994ccff050369720281">🤗 Hugging Face</a>
    &nbsp&nbsp | &nbsp&nbsp
    <a href="https://arxiv.org/pdf/2409.04774">📑 Paper</a>
    &nbsp&nbsp | &nbsp&nbsp
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/rgtjf/Untie-the-Knots&count_bg=#E97EBA&title_bg=#555555&icon=&icon_color=#E7E7E7&title=visitors&edge_flat=false" alt="Hits"></a>
</div>

## Introduction

We introduce Untie the Knots, a novel data augmentation strategy employed during the continue pre-training phase, designed to efficiently enable LLMs to gain long-context capabilities without the need to modify the existing data mixture. 

In particular, we chunk the documents, shuffle the chunks, and create a complex and knotted structure of long texts; LLMs are then trained to untie these knots and identify relevant segments within seemingly chaotic token sequences. This approach greatly improves the model's performance by accurately attending to relevant information in long context and the training efficiency is also largely increased. 

We conduct extensive experiments on models with 7B and 72B parameters, trained on 20 billion tokens, demonstrating that UtK achieves 75% and 84.5% accurracy on RULER at 128K context length, significantly outperforming other long context strategies. The trained models will open-source for further research.

## Performance
<img width="500" alt="image" src="https://github.com/user-attachments/assets/f0fb52f7-9c4d-45fc-95f7-88d3f24ddc21">



## License

The content of this project itself is licensed under [LICENSE](LICENSE).


## Citation

If you find this repo helpful, please cite our paper as follows:

```
@article{tian2024utk,
  title={Untie-the-Knots: An Efficient Data Augmentation Strategy for Long-Context Pre-Training in Language Models},
  author={Junfeng Tian, Da Zheng, Yang Chen, Rui Wang, Colin Zhang, Debing Zhang},
  journal={arXiv preprint arXiv:TODO},
  year={2024}
}
```

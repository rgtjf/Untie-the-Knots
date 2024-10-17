# Untie-the-Knots: An Efficient Data Augmentation Strategy for Long-Context Pre-Training in Language Models

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


## Model Details

Qwen2-UtK-7B-128K is a continuation of the Qwen2-7B model, incorporating RoPE theta modification (from 1,000,000 to 5,000,000). We also provide Qwen2-UtK-ChatQA2-7B-128K, trained with long SFT data from ChatQA 2.0 to enhance extended context handling. We also provide Qwen2-UtK-72B-128K and Qwen2-UtK-ChatQA2-72B-128K for further research.

## Long Text Processing

For deployment, we recommend using **vLLM**:

1. **Install vLLM**:
   ```bash
   pip install "vllm>=0.4.3"
   ```
   Or install from [source](https://github.com/vllm-project/vllm/).

2. **Deploy the Model**:
   ```bash
   python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-UtK-ChatQA2-7B-128K --model path/to/weights \
       --trust-remote-code --tensor-parallel-size 2 --host 0.0.0.0 --enable_chunked_prefill --max_num_batched_tokens 32768
   ```
   You can access the chat API using:
   ```bash
   curl http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
       "model": "Qwen2-UtK-ChatQA2-7B-128K",
       "messages": [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Your Long Input Here."}
       ]
       }'
   ```
   For 72B, please use `--tensor-parallel-size 8`.

## Evaluation

### Performance on RULER (Base Model)

![RULER Performance](https://github.com/user-attachments/assets/f0fb52f7-9c4d-45fc-95f7-88d3f24ddc21)

### Performance on InfiniteBench (Instruct Model)

| Model                              | En.Avg. | En.Sum | En.QA | En.MC | En.Dia | Zh.QA |
|------------------------------------|---------|--------|-------|-------|--------|-------|
| GPT-4-Turbo-2024-04-09             | 33.2    | 17.6   | 19.3  | 77.7  | 18.0   | -     |
| Claude 2                           | 34.0    | 14.5   | 12.0  | 62.9  | 46.5   | 9.6   |
| Kimi-Chat                          | 29.6    | 18.0   | 16.5  | 72.5  | 11.5   | 17.9  |
| Yi-34B-200K                        | < 15.15 | < 5    | 12.2  | 38.4  | <5     | 13.6  |
| Qwen2-72B-Instruct                 | 39.8    | 31.7   | 21.5  | 83.0  | 23.0   | -     |
| Llama-3-70B-Instruct-Gradient-262k | 32.6    | 14.3   | 29.5  | 69.0  | 17.5   | -     |
| Llama3.1-8B-Instruct               | 33.2    | 29.2   | 31.5  | 59.0  | 13.0   | -     |
| Llama3.1-70B-Instruct              | 39.8    | 30.9   | 38.5  | 75.6  | 14.3   | -     |
| Llama3-ChatQA-2-8B                 | 35.6    | 17.1   | 43.5  | 64.2  | 17.5   | -     |
| Llama3-ChatQA-2-70B                | 41.0    | 16.1   | 48.2  | 80.4  | 19.5   | -     |
| Qwen2-UtK-ChatQA2-7B-128K              | 33.3    | 21.2   | 42.6  | 61.1  | 8.5    | 37.6  |
| Qwen2-UtK-CHatQA2-72B-128K             | 47.3    | 18.2   | 55.9  | 83.8  | 31.0   | 45.2  |



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

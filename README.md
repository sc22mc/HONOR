<h1 align="center">
    MAGICGUI: A FOUNDATIONAL MOBILE GUI AGENT WITH SCALABLE DATA PIPELINE AND REINFORCEMENT FINE-TUNING
</h1>

<p align="center">
    【English | <a href="README_zh.md">中文</a>】
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="https://huggingface.co/openbmb/AgentCPM-GUI">Model</a> •
  <a href="#evaluation-data">Evaluation Data</a> •
  <a href="https://arxiv.org/abs/2506.01391">Technical Report</a>
</p>

## News

* [2025-07-20] 📄📄📄 We have released the **technical report** of AgentCPM-GUI! Check it out [here](https://arxiv.org/abs/2506.01391).
* [2025-07-20] 🚀🚀🚀 We have open-sourced **MAGICGUI**, an on-device GUI agent capable of operating Chinese & English apps and equipped with RFT-enhanced reasoning abilities.

## Overview

**AgentCPM-GUI** is an open-source on-device LLM agent model jointly developed by [THUNLP](https://nlp.csai.tsinghua.edu.cn), Renmin University of China and [ModelBest](https://modelbest.cn/en). Built on [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) with 8 billion parameters, it accepts smartphone screenshots as input and autonomously executes user-specified tasks. 

Key features include:

- **High-quality GUI grounding** — Pre-training on a large-scale bilingual Android dataset significantly boosts localization and comprehension of common GUI widgets (buttons, input boxes, labels, icons, etc.).
- **Chinese-app operation** — The first open-source GUI agent finely tuned for Chinese apps, covering 30 + popular titles such as Amap, Dianping, bilibili and Xiaohongshu.
- **Enhanced planning & reasoning** — Reinforcement fine-tuning (RFT) lets the model “think” before outputting an action, greatly improving success on complex tasks.
- **Compact action-space design** — An optimized action space and concise JSON format reduce the average action length to 9.7 tokens, boosting on-device inference efficiency.

Demo Case (1x speed):

https://github.com/user-attachments/assets/694d3c2c-12ce-4084-8feb-4937ca9ad247

## Quick Start

### Install dependencies

```bash
git clone https://github.com/OpenBMB/AgentCPM-GUI
cd AgentCPM-GUI
conda create -n gui_agent python=3.11
conda activate gui_agent
pip install -r requirements.txt
```

### Download the model

Download [AgentCPM-GUI](https://huggingface.co/openbmb/AgentCPM-GUI) from Hugging Face and place it in `model/AgentCPM-GUI`.

#### Huggingface Inference

```python
import torch
from utils.model import Qwen2VLChat

# 1. Load the model and tokenizer
model_path = "model/MAGICGUI"  # model path
model = Qwen2VLChat.from_pretrained(model_path, min_pixels=4*28*28, max_pixels=768*28*28)
model = model.to("cuda:0") 

# 2. Build the input
instruction = "请找出屏幕截图中的选项区，要求距离坐标点<point>(167,84)最近。注意，仅定位最相关的控件即可，以<|box_start|>矩形框<|box_end|>格式输出。输出示例：<|box_start|>(70,58)(125,86)<|box_end|>"
image_path = "./assets/test_img/grounding.png"

# 3. Build the message format
messages = [{"type": "image", "value":f"{image_path}",
            {"type": "text", "value":f"{instruction}"]

# 4. Inference
response = model.generate(
    message = messages,
)

print(response)
```

Expected output:

```JSON
{"<|box_start|>(48,92)(853,137)<|box_end|>"}
```

### Action Space

At each step, the agent outputs is a single JSON object that contains:
- One (and only one) primitive action, chosen from the list below;
- Optional modifiers (`duration`, `thought`) and/or a task-level flag (`STATUS`).

Note that all keywords are **case-sensitive**, and we use **compact JSON** (i.e., no extra whitespace), which affects the tokenizer’s behavior.

| Action         | Required field(s)                                                                                           | Optional field(s)             | Purpose                                                                     |  Example                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------ |
| **Click**               | `POINT:[x,y]`                                                                                               | `duration`,`thought`,`STATUS` | Single tap at the normalized screen coordinate (0–1000, origin = top-left). | `{"POINT":[480,320]}`                            |
| **Long Press**               | `POINT:[x,y]`<br>`duration:1000`                                                                                               | `duration`,`thought`,`STATUS` | Touch-and-hold at coordinate (set a longer duration, e.g. >200 ms). | `{"POINT":[480,320],"duration":1000}`                            |
| **Swipe**      | `POINT:[x,y]`<br>`to:"up" \| "down" \| "left" \| "right"` **or** `to:[x,y]`                                 | `duration`,`thought`,`STATUS` | Swipe from the start point toward a direction **or** another coordinate.     | `{"POINT":[500,200],"to":"down"}` |
| **Press key**         | `PRESS:"HOME" \| "BACK" \| "ENTER"`                                                                         | `duration`,`thought`,`STATUS` | Trigger a hardware / navigation button.                                     | `{"PRESS":"HOME"}`                |
| **Type text**         | `TYPE:"<text>"`                                                                    | `duration`,`thought`,`STATUS` | Insert the given text at the current input focus.                           | `{"TYPE":"Hello, world!"}`                       |
| **Wait**              | `duration`                                                                              | `thought`,`STATUS`            | Idle for the specified time without any other action.                       | `{"duration":500}`                               |
| **Task-level status** | `STATUS:"start" \| "continue" \| "finish" \| "satisfied" \| "impossible" \| "interrupt" \| "need_feedback"` | `duration`,`thought`          | Report task progress; may appear **alone** or **with a primitive action**.  | `{"STATUS":"finish"}`                        |


## Fine-tuning

Source code for SFT and RFT training is provided — see [SFT](sft/readme.md) and [RFT](rft/readme.md).

## Performance Evaluation

### Grounding Benchmark

| Model                   | Fun2Point | Text2Point | Bbox2text | Average |
|-------------------------|-----------|------------|-----------|--------|
| **AgentCPM-GUI-8B**     | **79.1**  | **76.5**   | **58.2**  |**71.3**|
| Qwen2.5-VL-7B           | 59.8      | 59.3       | <ins>50.0</ins>      | <ins>56.4</ins>   |
| Intern2.5-VL-8B         | 17.2      | 24.2       | 45.9      | 29.1   |
| Intern2.5-VL-26B        | 14.8      | 16.6       | 36.3      | 22.6   |
| OS-Genesis-7B	        | 8.3	      | 5.8	       | 4.0       | 6.0    |
| UI-TARS-7B              | 56.8      | <ins>66.7</ins>       | 1.4       | 41.6   |
| OS-Atlas-7B             | 53.6      | 60.7       | 0.4       | 38.2   |
| Aguvis-7B	              | <ins>60.8</ins>      | **76.5**   | 0.2       | 45.8   |
| GPT-4o                  | 22.1      | 19.9       | 14.3      | 18.8   |
| GPT-4o with Grounding   | 44.3      | 44.0       | 14.3      | 44.2   |

### Agent Benchmark

| Dataset                   | Android Control-Low TM | Android Control-Low EM | Android Control-High TM | Android Control-High EM | GUI-Odyssey TM  | GUI-Odyssey EM  | AITZ TM         | AITZ EM         | Chinese APP (CAGUI) TM  | Chinese APP (CAGUI) EM  |
| ------------------------- | ---------------------- | ---------------------- | ----------------------- | ----------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| **AgentCPM-GUI-8B** | <ins>94.39</ins> | <ins>90.20</ins> | <ins>77.70</ins> | <ins>69.17</ins> | **90.85** | **74.96** | **85.71** | **76.38** | **96.86** | **91.28** |
| Qwen2.5-VL-7B             | 94.14                  | 84.96                  | 75.10                   | 62.90                   | 59.54           | 46.28           | 78.41           | 54.61           | 74.18            | 55.16           |
| UI-TARS-7B                | **95.24**                  | **91.79**                  | **81.63**                   | **74.43**                   | 86.06           | 67.90           | <ins>80.42</ins>           | <ins>65.77</ins>           | <ins>88.62</ins>           | <ins>70.26</ins>           |
| OS-Genesis-7B             | 90.74                  | 74.22                  | 65.92                   | 44.43                   | 11.67           | 3.63            | 19.98           | 8.45            | 38.10           | 14.50           |
| OS-Atlas-7B               | 73.03                  | 67.25                  | 70.36                   | 56.53                   | 91.83*            | 76.76*           | 74.13           | 58.45           | 81.53           | 55.89           |
| Aguvis-7B                 | 93.85                  | 89.40                  | 65.56                   | 54.18                   | 26.71           | 13.54           | 35.71           | 18.99           | 67.43           | 38.20           |
| OdysseyAgent-7B           | 65.10                  | 39.16                  | 58.80                   | 32.74                   | <ins>90.83</ins>           | <ins>73.67</ins>           | 59.17           | 31.60           | 67.56           | 25.44           |
| GPT-4o                    | -                      | 19.49                  | -                       | 20.80                   | -               | 20.39           | 70.00           | 35.30           | 3.67            | 3.67            |
| Gemini 2.0                | -                      | 28.50                  | -                       | 60.20                   | -               | 3.27            | -               | -               | -               | -               |
| Claude                    | -                      | 19.40                  | -                       | 12.50                   | 60.90           | -               | -               | -               | -               | -               |

> \*Different train/test splits

TM and EM stand for the **Type Match** and **Exact Match**, respectively. All evaluation data and code are open-sourced — see [here](eval) for details.

## Evaluation Data

We provide **CAGUI**, an evaluation benchmark for Chinese apps covering **grounding** and **agent** tasks.
See the dataset on [Hugging Face](https://huggingface.co/datasets/openbmb/CAGUI).

## FAQs

Click here to view the [FAQs](https://github.com/OpenBMB/AgentCPM-GUI/blob/main/eval/README.md#faqs).

## Trends

<a href="https://star-history.com/#OpenBMB/AgentCPM-GUI&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=OpenBMB/AgentCPM-GUI&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=OpenBMB/AgentCPM-GUI&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=OpenBMB/AgentCPM-GUI&type=Date" />
 </picture>
</a>

## License

* Code in this repository is released under the [Apache-2.0](./LICENSE) license.

## Citation

If **AgentCPM-GUI** is useful for your research, please cite:

```bibtex
@article{zhang2025agentcpmgui,
      title={Agent{CPM}-{GUI}: Building Mobile-Use Agents with Reinforcement Fine-Tuning}, 
      author={Zhong Zhang and Yaxi Lu and Yikun Fu and Yupeng Huo and Shenzhi Yang and Yesai Wu and Han Si and Xin Cong and Haotian Chen and Yankai Lin and Jie Xie and Wei Zhou and Wang Xu and Yuanheng Zhang and Zhou Su and Zhongwu Zhai and Xiaoming Liu and Yudong Mei and Jianming Xu and Hongyan Tian and Chongyi Wang and Chi Chen and Yuan Yao and Zhiyuan Liu and Maosong Sun},
      year={2025},
      journal={arXiv preprint arXiv:2506.01391},
}
```

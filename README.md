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

| Action             | Description                                                             | Conditions for R<sub>acc</sub> = +2                                                                              |
|--------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **Tap**            | Click at coordinate (x, y)                                               | dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%                                                               |
| **Scroll**         | Scroll at coordinate (x, y) with direction up / down / left / right     | dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14% and direction = gt[direction]                                |
| **Text Input**     | Type *text* at coordinate (x, y)                                         | dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14% and F1(text, gt[text]) > 0.5                                 |
| **Navigation Back**| Adb command to go back to the previous page                             | –                                                                                                                |
| **Navigation Home**| Adb command to go to the home screen of the mobile                      | –                                                                                                                |
| **Long Press**     | Long Press at coordinate (x, y)                                          | dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%                                                               |
| **Finish**         | Indicate that navigate task has been completed                          | –                                                                                                                |
| **Wait**           | wait for several seconds                                                 | –                                                                                                                |
| **Enter**          | Adb command to press enter                                               | –                                                                                                                |
| **Takeover**       | Request user takeover                                                    | –                                                                                                                |
| **Drag**           | Drag from coordinate (x₁, y₁) to coordinate (x₂, y₂)                    | dist([x₁, y₁], [x<sub>1c</sub>, y<sub>1c</sub>]) ≤ 7.5% and dist([x₂, y₂], [x<sub>2c</sub>, y<sub>2c</sub>]) ≤ 7.5% |
| **Call API**       | Adb command to *open/kill* app                                           | app = gt[app] and open/kill = gt[operation]                                                                      |
| **Screenshot**     | Adb command to screenshot                                                | –                                                                                                                |
| **Long Screenshot**| Adb command to long screenshot                                           | –                                                                                                                |



## Fine-tuning

Source code for SFT and RFT training is provided — see [SFT](sft/readme.md) and [RFT](rft/readme.md).

## Performance Evaluation

### Performance comparison on the Magic-RICH dataset

| Model                | <th colspan="3">Routine</th> | <th colspan="3">Instruction</th> | <th colspan="3">Complex</th> | <th colspan="3">Handling</th> | Exception |
|----------------------|------------------------------|----------------------------------|------------------------------|--------------------------------|-----------|
|                      | Type | Grd | SR | Type | Grd | SR | Type | Grd | SR | Type | Grd | SR | – |
| **GPT-4o**            | 49.3 | 16.7 | 4.6 | 56.6 | 13.5 | 19.8 | 49.0 | 14.6 | 7.4 | 85.1 | – | – | 70.8 |
| **Gemini 2.0**        | 89.2 | 49.4 | 34.7 | 84.1 | 54.2 | 51.4 | 83.3 | 50.3 | 42.0 | 73.7 | – | – | 68.3 |
| **InternVL-2-8B**     | 30.1 | 2.8  | 1.3 | 37.1 | 4.0  | 15.8 | 17.1 | 6.0  | 1.3 | 70.8 | – | – | 68.3 |
| **Qwen2-VL-7B**       | 71.7 | 41.0 | 28.1 | 73.6 | 43.9 | 41.5 | 65.6 | 28.7 | 21.2 | 67.0 | – | – | 67.0 |
| **Qwen2.5-VL-7B**     | 94.3 | 92.6 | 76.3 | 89.3 | 95.7 | 83.6 | 86.6 | 69.6 | 60.0 | 67.0 | – | – | 67.0 |
| **UI-TARS-7B**        | 83.5 | 84.9 | 73.3 | 76.6 | 85.6 | 69.8 | 91.4 | 69.1 | 67.0 | 3.6  | – | – | 67.0 |
| **UI-TARS-1.5-7B**    | 85.6 | 96.2 | 81.5 | 78.6 | 92.1 | 72.2 | 94.7 | 74.3 | 71.1 | 57.0 | – | – | 57.0 |
| **MiMo-VL-7B-SFT**    | 93.0 | 77.9 | 65.3 | 89.7 | 85.7 | 75.4 | 89.1 | 80.1 | 71.0 | 67.0 | – | – | 67.0 |
| **AgentCPM-GUI**      | 84.3 | 92.2 | 75.1 | 70.4 | 80.7 | 56.0 | 72.3 | 54.6 | 39.4 | 74.1 | – | – | 74.1 |
| **MagicGUI-CPT**      | 98.5 | 98.5 | 97.2 | 95.5 | 96.3 | 92.9 | 88.5 | 82.3 | 72.9 | 93.2 | – | – | 93.2 |
| **MagicGUI-RFT**      | 99.7 | 97.5 | 97.5 | 97.2 | 95.6 | 94.0 | 92.1 | 80.4 | 74.1 | 92.1 | – | – | 92.1 |





### Performance comparison on open-source AndroidControl and GUI-Odyssey datasets. 

| Model                  | AC-Low | AC-High | GUI-Odyssey |
|------------------------|--------|---------|-------------|
| **GPT-4o**              | -      | 19.5    | -           |
| **Gemini 2.0**          | -      | 28.5    | -           |
| **Claude 2.0**          | -      | 28.5    | -           |
| **Qwen2-VL-7B**         | 55.7   | 45.8    | 58.6        |
| **Qwen2.5-VL-7B**       | 94.1   | 75.1    | 59.5        |
| **Aguvis-7B**           | 93.9   | 65.6    | 26.7        |
| **OS-Atlas-7B**         | 73.0   | 70.4    | 91.8*       |
| **UI-TARS-7B**          | 95.2   | 81.6    | 86.1        |
| **AgentCPM-GUI**        | 94.4   | 77.7    | 90.9        |
| **MagicGUI-CPT**        | 97.2   | 94.5    | 90.4        |
| **MagicGUI-RFT**        | 99.7   | 93.5    | 89.7        |


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

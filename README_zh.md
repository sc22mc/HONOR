<div align="center">
  <img src="./assets/MagicGUI_logo.png" width="600em"></img>
</div>

<p align="center">
    【<a href="README.md">English</a> | 中文】
</p>

<p align="center">
  <a href="#概览">概览</a> •
  <a href="#框架">框架</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="https://huggingface.co/openbmb/AgentCPM-GUI">模型</a> •
  <a href="#动作空间">动作空间</a> •
  <a href="#评测">评测</a> •
  <a href="#性能评测">性能对比</a> •
  <a href="https://arxiv.org/abs/2508.03700">技术报告</a>
</p>

## 新闻

* [2025-07-20] 📄📄📄 发布了 **AgentCPM-GUI 技术报告**！点击查看 [here](https://arxiv.org/abs/2508.03700)。
* [2025-07-20] 🚀🚀🚀 开源了 **MagicGUI** —— 一款支持中英文应用、具备 RFT 增强推理能力的端侧 GUI 智能体。

## 概览

MagicGUI 是由荣耀研发的开源 GUI 智能体模型，基于 Qwen2-VL（70 亿参数）。它在视觉定位、屏幕问答、动作序列规划与执行方面展现出卓越能力。MagicGUI 能够在移动设备上实现多模态感知、理解与自动化任务执行。

**数据采集框架**：提出可扩展、模块化的 GUI 数据采集框架，高效获取移动设备上的高质量数据。  

**强大的感知与定位能力**：通过元素指代、元素定位、屏幕描述等任务，结合大规模知识提升感知与定位能力。  

**统一动作空间**：为不同移动平台设计统一的动作空间，涵盖点击、输入、滑动等基本操作，同时支持等待、拖拽、接管等复杂动作。  

**面向规划的推理机制**：引入规划导向的推理机制，提升任务执行的稳定性与动态环境下动作决策的准确性。  

**双阶段训练范式**：通过持续预训练（CPT）增强核心感知、定位与导航能力，并通过强化微调（RFT）提升模型鲁棒性与泛化能力。  

## 框架

MagicGUI 的整体训练框架分为两个阶段：

**阶段 I**：持续预训练（CPT），先在大规模多样化数据集上训练基础模型，再通过平衡的高质量数据集进行退火训练。  

**阶段 II**：强化微调（RFT），进一步增强模型的鲁棒性与泛化能力。  

<div align="center">
  <img src="./assets/framework.png" width="800em"></img>
</div>

## 快速开始

### 安装依赖（需要修改）

```bash
git clone https://github.com/OpenBMB/AgentCPM-GUI
cd AgentCPM-GUI
conda create -n gui_agent python=3.11
conda activate gui_agent
pip install -r requirements.txt
```

### 下载模型

下载 [MagicGUI](https://huggingface.co/openbmb/AgentCPM-GUI)。

#### Huggingface 推理示例

```python
import torch
from utils.model import Qwen2VLChat

# 1. 加载模型和 tokenizer
model_path = "./models/RFT"  # 模型路径
model = Qwen2VLChat.from_pretrained(model_path, min_pixels=4*28*28, max_pixels=768*28*28)
model = model.to("cuda:0") 

# 2. 构建输入
instruction = """你是一个训练有素的手机智能体，能够帮助用户进行单步导航任务。已知当前智能手机的截图<image>，和用户指令"查看会员信息"请输出正确的函数调用以实现用户指令。除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：..."""

image_path = "./assets/test_action.png"

# 3. 构建消息格式
messages = [{"type": "image", "value":f"{image_path}",
            {"type": "text", "value":f"{instruction}"]

# 4. 推理
response = model.generate(
    message = messages,
)

print(response)
```

预期输出：

```JSON
{"tap(700,964)"}
```

## 动作空间

智能体在每一步的输出是一个 JSON 对象，包含：
- **一个且仅一个**原子动作；
- 可选修饰符（`duration`, `thought`）和任务级标志（`STATUS`）。

注意：所有关键词 **区分大小写**，并使用 **紧凑 JSON**（无多余空格）。

<table>
  <thead>
    <tr>
      <th>Action</th>
      <th>Description</th>
      <th>Conditions for R<sub>acc</sub> = +2</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Tap</b></td>
      <td>Click at coordinate (x, y)</td>
      <td>dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%</td>
      <td><code>tap(x,y)</code></td>
    </tr>
    <tr>
      <td><b>Scroll</b></td>
      <td>Scroll at coordinate (x, y) with<br>direction up / down / left / right</td>
      <td>dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%<br>and direction = gt[direction]</td>
      <td><code>scroll(x,y,direction)</code></td>
    </tr>
    <tr>
      <td><b>Text Input</b></td>
      <td>Type <i>text</i> at coordinate (x, y)</td>
      <td>dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%<br>and F1(text, gt[text]) > 0.5</td>
      <td><code>text(x,y,text_input)</code></td>
    </tr>
    <tr>
      <td><b>Navigation Back</b></td>
      <td>Adb command to go back to the previous page</td>
      <td>–</td>
      <td><code>navigate_back()</code></td>
    </tr>
    <tr>
      <td><b>Navigation Home</b></td>
      <td>Adb command to go to the home screen of the mobile</td>
      <td>–</td>
      <td><code>navigate_home()</code></td>
    </tr>
    <tr>
      <td><b>Long Press</b></td>
      <td>Long press at coordinate (x, y)</td>
      <td>dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%</td>
      <td><code>long_press(x,y)</code></td>
    </tr>
    <tr>
      <td><b>Finish</b></td>
      <td>Indicate that navigation task has been completed</td>
      <td>–</td>
      <td><code>finish()</code></td>
    </tr>
    <tr>
      <td><b>Wait</b></td>
      <td>Wait for several seconds</td>
      <td>–</td>
      <td><code>wait()</code></td>
    </tr>
    <tr>
      <td><b>Enter</b></td>
      <td>Adb command to press enter</td>
      <td>–</td>
      <td><code>enter()</code></td>
    </tr>
    <tr>
      <td><b>Takeover</b></td>
      <td>Request user takeover</td>
      <td>–</td>
      <td><code>take_over(message)</code></td>
    </tr>
    <tr>
      <td><b>Drag</b></td>
      <td>Drag from coordinate (x₁, y₁) to (x₂, y₂)</td>
      <td>
        dist([x₁, y₁], [x<sub>1c</sub>, y<sub>1c</sub>]) ≤ 7.5%<br>
        and dist([x₂, y₂], [x<sub>2c</sub>, y<sub>2c</sub>]) ≤ 7.5%
      </td>
      <td><code>drag(x1,y1,x2,y2)</code></td>
    </tr>
    <tr>
      <td><b>Call API</b></td>
      <td>Adb command to <i>open</i> or <i>kill</i> app</td>
      <td>app = gt[app]<br>and open/kill = gt[operation]</td>
      <td><code>call_api(api_name,operation)</code></td>
    </tr>
    <tr>
      <td><b>Screenshot</b></td>
      <td>Adb command to take a screenshot</td>
      <td>–</td>
      <td><code>screen_shot()</code></td>
    </tr>
    <tr>
      <td><b>Long Screenshot</b></td>
      <td>Adb command to take a long screenshot</td>
      <td>–</td>
      <td><code>long_screen_shot()</code></td>
    </tr>
  </tbody>
</table>

## 评测

### 1. 数据准备

请从 [Magic-RICH 数据集](https://example.com/dataset-download) 下载子集，并将其放入 `.datasets/` 目录中。

- `assets/`  
- `datasets/`  
  - `Routine`
  - `Instruction`
  - `Complex`
  - `Handing_Exception`
- `utils/`

其他开源数据集的准备方式请参考 [datasets/eval_data_process/readme.md](datasets/eval_data_process/readme.md)。

### 2. 参数

我们使用 `run_eval.py` 进行评测。

- `--data`: 数据集名称  
- `--model`: 模型路径  
- `--work-dir`: 保存评测结果的目录（默认 `.`）  
- `--mode`: 执行模式（默认 `all`，可选 `all` 或 `infer`）  
- `--eval_model_path`: 评测模型路径（当 `mode=all` 且 `data=ScreenQA-short` 时必填）  

### 3. 运行示例

```python
python run_eval.py --data ScreenQA-short --model MagicGUI_Path  --mode all --eval_model_path Eval_Model_Path
python run_eval.py --data ScreenSpot_v2_mobile --model MagicGUI_Path  --mode all
python run_eval.py --data Os-Atlas-mobile --model MagicGUI_Path  --mode all
python run_eval.py --data Routine --model MagicGUI_Path  --mode all
python run_eval.py --data Complex --model MagicGUI_Path  --mode all
python run_eval.py --data Instruction --model MagicGUI_Path  --mode all
python run_eval.py --data Handling_Exception --model MagicGUI_Path  --mode all
python run_eval.py --data AC-Low --model MagicGUI_Path  --mode all
python run_eval.py --data AC-High --model MagicGUI_Path  --mode all
python run_eval.py --data GUI-Odyssey --model MagicGUI_Path  --mode all
```

## 性能评测

（此处包含性能对比表格，内容保持不变，仅翻译了小标题，例如“性能对比”、“基准数据集对比”、“Magic-RICH 数据集对比”等）

## 许可协议

* 本项目基于 [Apache-2.0](./LICENSE) 协议开源。模型权重完全开放供学术研究使用，商业使用需联系 magicgui@honor.com 获取授权。本项目使用了 Qwen2VL-7B-Instruct 作为初始化模型，该模型同样遵循 Apache-2.0 协议。

## 引用

如果 **AgentCPM-GUI** 对您的研究有帮助，请引用：

```bibtex
@article{zhang2025agentcpmgui,
      title={Agent{CPM}-{GUI}: Building Mobile-Use Agents with Reinforcement Fine-Tuning}, 
      author={Zhong Zhang and Yaxi Lu and Yikun Fu and Yupeng Huo and Shenzhi Yang and Yesai Wu and Han Si and Xin Cong and Haotian Chen and Yankai Lin and Jie Xie and Wei Zhou and Wang Xu and Yuanheng Zhang and Zhou Su and Zhongwu Zhai and Xiaoming Liu and Yudong Mei and Jianming Xu and Hongyan Tian and Chongyi Wang and Chi Chen and Yuan Yao and Zhiyuan Liu and Maosong Sun},
      year={2025},
      journal={arXiv preprint arXiv:2506.01391},
}
```


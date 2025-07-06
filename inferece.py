import torch
from .utils.model import Qwen2VLChat

# 1.Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chaimingxu-240108540141/HONOR_Test/model/Qwen2-VL-7B-Instruct"
model = Qwen2VLChat.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device) 

# 2.Build the input
instruction = "请找出屏幕截图中的选项区，要求距离坐标点<point>(167,84)最近。注意，仅定位最相关的控件即可，以<|box_start|>矩形框坐标<|box_end|>格式输出。输出示例：<|box_start|>(70,58),(125,86)<|box_end|>"
image_path = "./assets/test_img/grounding.png"
messages = [{"type": "image", "value": f"{image_path}"}, 
          {"type": "text", "value": f"{instruction}"}]

# 3.Inference
response = model.generate(
    messages = messages,
)

print(response)
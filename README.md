import os
import json
import shutil

def process_jsonl_and_images(jsonl_path, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)
    
    # 新的jsonl文件路径
    new_jsonl_path = os.path.join(target_folder, os.path.basename(jsonl_path))
    
    # 第一步：读取奇数行（1开始计数）
    new_lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile, 1):
            if i % 2 == 1:
                new_lines.append(line)

    # 写入新的 jsonl 文件
    with open(new_jsonl_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(new_lines)
    
    # 第二步：创建 images 文件夹
    images_folder = os.path.join(target_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)
    
    # 第三步：复制图片并修改jsonl中的路径
    updated_lines = []
    for line in new_lines:
        data = json.loads(line)
        updated_images = []
        for img_path in data.get('images', []):
            filename = os.path.basename(img_path)
            target_img_path = os.path.join(images_folder, filename)
            if os.path.exists(img_path):
                shutil.copy2(img_path, target_img_path)
            updated_images.append(filename)
        data['images'] = updated_images
        updated_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 最后一步：覆盖更新过的 jsonl 文件
    with open(new_jsonl_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(updated_lines)

    print(f"处理完成，新的 jsonl 文件保存在：{new_jsonl_path}")


if 'image' in line:
    tgt_path = [os.path.join(base_dir, line['image'])]
elif 'images' in line:
    tgt_path = [os.path.join(base_dir, img) for img in line['images']]

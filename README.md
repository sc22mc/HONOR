if 'image' in line:
                tgt_path = [line['image']]
            elif 'images' in line:
                tgt_path = line['images']


if 'image' in line:
                tgt_path = os.path.join(f'{os.path.dirname(self.processed_path)}',[line['image']])
            elif 'images' in line:
                tgt_path = os.path.join(f'{os.path.dirname(self.processed_path)}',line['images'])

# Data Processing Scripts

```
# Setup environment

cd MagicGUI/datasets/eval_data_process
conda create -n process_data python=3.11
conda activate process_data
pip install -r requirements.txt

mkdir tmp && cd tmp
git clone https://github.com/deepmind/android_env/
cd android_env; pip install .
```

## Android Control

Download [Android Control](https://github.com/google-research/google-research/tree/master/android_control) and save at ``MagicGUI/datasets/eval_data_process/tmp/android_control`` 
- `tmp/`: 
  - `android_control/`：
```
cd MagicGUI/datasets/eval_data_process
python process_android_control.py
```

## gui-odyssey

Download [GUI-Odyssey](https://github.com/OpenGVLab/GUI-Odyssey?tab=readme-ov-file) and save at ``MagicGUI/datasets/eval_data_process/tmp/GUI-Odyssey``. Copy [preprocessing.py](https://github.com/OpenGVLab/GUI-Odyssey/blob/master/data/preprocessing.py) and [format_converter.py](https://github.com/OpenGVLab/GUI-Odyssey/blob/master/data/format_converter.py) from the GUI-Odyssey repo to ``MagicGUI/datasets/eval_data_process/tmp/GUI-Odyssey``
- `tmp/`: 
  - `GUI-Odyssey/`：
```
cd MagicGUI/datasets/eval_data_process/tmp/GUI-Odyssey
python preprocessing.py
python format_converter.py
python ../../process_odyssey.py
```

## ScreenQA-short

Download [test subset of ScreenQA-short](https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA-Short/tree/main/data) and save at ``MagicGUI/datasets/eval_data_process/tmp/ScreenQA-short``
- `tmp/`: 
  - `ScreenQA-short/`：
    - `test-00000-of-00002.parquet`：
    - `test-00001-of-00002.parquet`：
```
cd MagicGUI/datasets/eval_data_process
python process_screenqa.py
```

## ScreenSpot_v2_mobile

Download [ScreenSpot_v2_mobile](https://huggingface.co/datasets/HongxinLi/ScreenSpot_v2) and save at ``MagicGUI/datasets/eval_data_process/tmp/ScreenSpot_v2_mobile``

```
cd MagicGUI/datasets/eval_data_process
python process_screenspotv2.py
```

## Os-Atlas-mobile

Download [Os-Atlas-mobile-aw_mobile.json](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data/blob/main/mobile_domain/aw_mobile.json) and [Os-Atlas-mobile-images.zip](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data/blob/main/mobile_domain/mobile_images.zip) at ``MagicGUI/datasets/eval_data_process/tmp/Os-Atlas-mobile``, and unzip Os-Atlas-mobile-images.zip.

```
cd MagicGUI/datasets/eval_data_process
python process_screenspotv2.py
```

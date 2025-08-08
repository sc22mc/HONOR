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

```
cd MagicGUI/datasets/eval_data_process
python process_android_control.py
```

## gui-odyssey

Download [GUI-Odyssey](https://github.com/OpenGVLab/GUI-Odyssey?tab=readme-ov-file) and save at ``MagicGUI/datasets/eval_data_process/tmp/GUI-Odyssey``. Copy [preprocessing.py](https://github.com/OpenGVLab/GUI-Odyssey/blob/master/data/preprocessing.py) and [format_converter.py](https://github.com/OpenGVLab/GUI-Odyssey/blob/master/data/format_converter.py) from the GUI-Odyssey repo to ``MagicGUI/datasets/eval_data_process/tmp/GUI-Odyssey``

```
cd MagicGUI/datasets/eval_data_process/tmp/GUI-Odyssey
python preprocessing.py
python format_converter.py
python ../../process_odyssey.py
```

## ScreenQA-short

Download [test_subset of ScreenQA-short](https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA-Short/tree/main/data) and save at ``MagicGUI/datasets/eval_data_process/tmp/ScreenQA-short``

```
cd MagicGUI/datasets/eval_data_process
python process_android_control.py
```

from ..screen_understanding import *
REFERRING_CONFIGS = { }

GROUNDING_CONFIGS = {
        'ScreenSpot_v2_mobile': {
            'path': '/opt/nas/p/mm/ie_env/zhangshenghui/project/VLMEvalKit_new/test_files/screen_spot/v2/ScreenSpot_v2_mobile.jsonl',
            'type': 'GROUNDING',
            'prompt_template': '根据英语短语的指令要求，找到屏幕截图中能够完成指令要求的控件位置，返回对应的bbox框，格式如<|box_start|>(669, 515),(902, 538)<|box_end|>。当前指令为：{query}',
            'need_prompt_template': 'always',
            # 'data_format': 'POINT',
        },
        'os-atlas-aw_mobile': { # !!!!!!!!!!!!!!!!!!!缺失
            'path': '/opt/nas/p/mm/ie_env/zhangshenghui/project/VLMEvalKit_new/test_files/os-atlas/os-atlas-aw_mobile.jsonl',
            'type': 'GROUNDING',
            'prompt_template': '<image>\nAccording to the following instruction or description, find the object in the figure that can complete the instructions or meet the description, and return the corresponding bbox, the format is such as <|box_start|>(56, 114),(250, 123)<|box_end|>. Instructions or description: {query}',
            'need_prompt_template': 'always',
            # 'data_format': 'POINT',
        },
        'one_grounding': { # !!!!!!!!!!!!!!!!!!!缺失
            'path': '/opt/nas/p/mm/ie_env/zhangshenghui/project/VLMEvalKit_new/test_files/one_grounding/Agent_Data_Referring_Gronuding_v202_up_down_sample_train-one_grounding_dedup.jsonl',
            'type': 'GROUNDING',
            # 'data_format': 'POINT',
        },
        'Agent_Data_QA_grounding': { # !!!!!!!!!!!!!!!!!!!缺失
            'path': '/opt/nas/p/mm/ie_env/zhangshenghui/project/VLMEvalKit_new/test_files/Agent_Data_QA/Agent_Data_QA_v300_clear_internvl_grounding_dedup.jsonl',
            'type': 'GROUNDING',
            'prompt_template': '根据指令要求，找到屏幕截图中能够完成指令要求的控件位置，返回对应的bbox框，格式如<|box_start|>(669, 515),(902, 538)<|box_end|>。当前指令为： {query}',
            # 'prompt_template': '请找出屏幕截图，指令要求点击的控件位置，以<point>(点坐标)</point>格式输出。当前指令为： {query}',
            # 'prompt_template': '根据指令要求，找到屏幕截图中能够完成指令要求的控件位置，返回对应的bbox框，格式如<box>[[669, 515, 902, 538]]</box>。当前指令为： {query}',
            # 'prompt_template': '根据指令要求，找到屏幕截图中能够完成指令要求的控件位置，返回对应的bbox框，格式如<|det|>[[669, 515, 902, 538]]<|/det|>。当前指令为： {query}',
            'need_prompt_template': 'always',
            # 'data_format': 'POINT',
        },
    }

NAVIGATION_CONFIGS = {}

REFERRING_CONFIGS = {}
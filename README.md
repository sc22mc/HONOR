(cmx) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/VLMEvalKit# lmdeploy serve api_server /opt/nas/p/mm/ie_env/VLMEvalKit/models/Qwen2-VL
-7B-Instruct --server-port 23333
2025-07-08 11:19:52,247 - lmdeploy - WARNING - archs.py:51 - Fallback to pytorch engine because `/opt/nas/p/mm/ie_env/VLMEvalKit/models/Qwen2-VL-7B-Instruct` not supported by turbomind engine.
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
2025-07-08 11:20:26,869 - lmdeploy - WARNING - __init__.py:10 - Disable DLSlime Backend
Loading weights from safetensors: 100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:16<00:00,  3.32s/it]
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
INFO:     Started server process [3602229]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)
INFO:     127.0.0.1:48632 - "GET / HTTP/1.1" 200 OK

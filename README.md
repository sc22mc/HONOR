(cmx) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/VLMEvalKit# lmdeploy serve api_server /opt/nas/p/mm/ie_env/VLMEvalKit/models/Qwen2-VL-7B-Instruct-fp16 --server-port 23333
2025-07-08 14:12:38,320 - lmdeploy - WARNING - archs.py:51 - Fallback to pytorch engine because `/opt/nas/p/mm/ie_env/VLMEvalKit/models/Qwen2-VL-7B-Instruct-fp16` not supported by turbomind engine.
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
2025-07-08 14:12:55,788 - lmdeploy - WARNING - __init__.py:10 - Disable DLSlime Backend
Loading weights from safetensors:   0%|                                                                                | 0/4 [00:00<?, ?it/s]
Process mp_engine_proc:
Traceback (most recent call last):
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/engine/mp_engine/mp_engine.py", line 146, in _mp_proc
    engine = Engine.from_pretrained(
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/engine/engine.py", line 418, in from_pretrained
    return cls(model_path=pretrained_model_name_or_path,
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/engine/engine.py", line 359, in __init__
    self.executor.init()
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/engine/executor/base.py", line 177, in init
    self.build_model()
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/engine/executor/uni_executor.py", line 56, in build_model
    self.model_agent.build_model()
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/engine/model_agent.py", line 807, in build_model
    self._build_model()
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/engine/model_agent.py", line 798, in _build_model
    load_model_weights(patched_model, model_path, device=device)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/weight_loader/model_weight_loader.py", line 170, in load_model_weights
    loader.load_model_weights(model, device=device)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/weight_loader/model_weight_loader.py", line 161, in load_model_weights
    model.load_weights(weights_iterator)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/pytorch/models/qwen2_vl.py", line 784, in load_weights
    param = params_dict[name]
KeyError: 'model.language_model.layers.16.input_layernorm.weight'

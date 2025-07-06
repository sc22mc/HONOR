(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# /opt/nas/p/conda/envs/torch_zhangsh/bin/python /opt/nas/p/mm/ie_env/HONOR/inferece.py
Traceback (most recent call last):
  File "/opt/nas/p/mm/ie_env/HONOR/inferece.py", line 6, in <module>
    model = Qwen2VLChat(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
TypeError: Qwen2VLChat.__init__() got an unexpected keyword argument 'trust_remote_code'
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# /opt/nas/p/conda/envs/torch_zhangsh/bin/python /opt/nas/p/mm/ie_env/HONOR/inferece.py
Traceback (most recent call last):
  File "/opt/nas/p/conda/envs/torch_zhangsh/lib/python3.10/site-packages/transformers/utils/hub.py", line 342, in cached_file
    resolved_file = hf_hub_download(
  File "/opt/nas/p/conda/envs/torch_zhangsh/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 106, in _inner_fn
    validate_repo_id(arg_value)
  File "/opt/nas/p/conda/envs/torch_zhangsh/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 154, in validate_repo_id
    raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/opt/nas/p/mm/ie_env/HONOR/model/Qwen2-VL-7B'. Use `repo_type` argument if needed.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/nas/p/mm/ie_env/HONOR/inferece.py", line 6, in <module>
    model = Qwen2VLChat(model_path, min_pixels=4*28*28, max_pixels=768*28*28)
  File "/opt/nas/p/mm/ie_env/HONOR/utils/model/model.py", line 113, in __init__
    self.processor = Qwen2VLProcessor.from_pretrained(default_path)
  File "/opt/nas/p/conda/envs/torch_zhangsh/lib/python3.10/site-packages/transformers/processing_utils.py", line 1070, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
  File "/opt/nas/p/conda/envs/torch_zhangsh/lib/python3.10/site-packages/transformers/processing_utils.py", line 1116, in _get_arguments_from_pretrained
    args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
  File "/opt/nas/p/conda/envs/torch_zhangsh/lib/python3.10/site-packages/transformers/models/auto/image_processing_auto.py", line 453, in from_pretrained
    raise initial_exception
  File "/opt/nas/p/conda/envs/torch_zhangsh/lib/python3.10/site-packages/transformers/models/auto/image_processing_auto.py", line 440, in from_pretrained
    config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
  File "/opt/nas/p/conda/envs/torch_zhangsh/lib/python3.10/site-packages/transformers/image_processing_base.py", line 341, in get_image_processor_dict
    resolved_image_processor_file = cached_file(
  File "/opt/nas/p/conda/envs/torch_zhangsh/lib/python3.10/site-packages/transformers/utils/hub.py", line 408, in cached_file
    raise EnvironmentError(
OSError: Incorrect path_or_model_id: '/opt/nas/p/mm/ie_env/HONOR/model/Qwen2-VL-7B'. Please provide either the path to a local folder or the repo_id of a model on the Hub.
(torch_zhangsh) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/HONOR/utils/model# 

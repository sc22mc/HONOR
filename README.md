(cmx) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/VLMEvalKit# /opt/nas/p/conda/envs/cmx/bin/python /opt/nas/p/mm/ie_env/VLMEvalKit/download.py
Traceback (most recent call last):
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/utils/_http.py", line 409, in hf_raise_for_status
    response.raise_for_status()
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/requests/models.py", line 1026, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/config.json

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/transformers/utils/hub.py", line 470, in cached_files
    hf_hub_download(
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 1008, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 1115, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 1645, in _raise_on_head_call_error
    raise head_call_error
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 1533, in _get_metadata_or_catch_error
    metadata = get_hf_file_metadata(
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 1450, in get_hf_file_metadata
    r = _request_wrapper(
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 286, in _request_wrapper
    response = _request_wrapper(
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 310, in _request_wrapper
    hf_raise_for_status(response)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/huggingface_hub/utils/_http.py", line 482, in hf_raise_for_status
    raise _format(HfHubHTTPError, str(e), response) from e
huggingface_hub.errors.HfHubHTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/config.json (Request ID: Root=1-686cb490-792294054001de6a39db0fbb;fb73d5fa-1a99-48c1-86c0-2992918041d7)

Invalid credentials in Authorization header

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/nas/p/mm/ie_env/VLMEvalKit/download.py", line 3, in <module>
    model = Qwen2VLForConditionalGeneration.from_pretrained(
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/transformers/modeling_utils.py", line 309, in _wrapper
    return func(*args, **kwargs)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/transformers/modeling_utils.py", line 4323, in from_pretrained
    config, model_kwargs = cls.config_class.from_pretrained(
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/transformers/configuration_utils.py", line 555, in from_pretrained
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/transformers/configuration_utils.py", line 595, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/transformers/configuration_utils.py", line 654, in _get_config_dict
    resolved_config_file = cached_file(
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/transformers/utils/hub.py", line 312, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/transformers/utils/hub.py", line 553, in cached_files
    raise OSError(f"There was a specific connection error when trying to load {path_or_repo_id}:\n{e}")
OSError: There was a specific connection error when trying to load Qwen/Qwen2-VL-7B-Instruct:
401 Client Error: Unauthorized for url: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/config.json (Request ID: Root=1-686cb490-792294054001de6a39db0fbb;fb73d5fa-1a99-48c1-86c0-2992918041d7)

Invalid credentials in Authorization header
(cmx) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/VLMEvalKit# 

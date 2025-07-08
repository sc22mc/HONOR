(cmx) root@mm-fudan-chai-l20-1-0:/opt/nas/p/mm/ie_env/VLMEvalKit# lmdeploy convert hf /opt/nas/p/mm/ie_env/VLMEvalKit/models/Qwen2-VL-7B-Instruct --dst-path /opt/nas/p/mm/ie_env/VLMEvalKit/models/Qwen2-VL-7B-Instruct_fp16 --dtype float16
Traceback (most recent call last):
  File "/opt/nas/p/conda/envs/cmx/bin/lmdeploy", line 8, in <module>
    sys.exit(run())
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/cli/entrypoint.py", line 39, in run
    args.run(args)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/cli/cli.py", line 143, in convert
    main(**kwargs)
  File "/opt/nas/p/conda/envs/cmx/lib/python3.10/site-packages/lmdeploy/turbomind/deploy/converter.py", line 292, in main
    assert is_supported(model_path), (f'turbomind does not support {model_path}. '
AssertionError: turbomind does not support /opt/nas/p/mm/ie_env/VLMEvalKit/models/Qwen2-VL-7B-Instruct. Plz try pytorch engine instead.

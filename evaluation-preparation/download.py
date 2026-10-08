from huggingface_hub import snapshot_download, hf_hub_download
import pathlib,zipfile
snapshot_download('estjtmi/behavior-2026-five-winner-ckpt3-10k',revision='674056b95708df9d7a9e47faef84c9187f5014d0',local_dir='/workspace/checkpoints/ckpt3-10k',allow_patterns=['params/**','assets/**','_CHECKPOINT_METADATA'],max_workers=4)
for name,filename in [('behavior-1k-assets','behavior-1k-assets-3.9.0.zip'),('omnigibson-robot-assets','omnigibson-robot-assets-3.8.2.zip'),('2026-challenge-task-instances','2026-challenge-task-instances.zip')]:
 p=hf_hub_download('behavior-1k/zipped-datasets',filename,repo_type='dataset',local_dir='/workspace/asset-downloads')
 target=pathlib.Path('/workspace/BEHAVIOR-2026/datasets')/name
 target.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(p) as z:z.extractall(target)
 pathlib.Path(p).unlink()
 print('Extracted',name,flush=True)

"""Sequential stock-evaluator orchestration; exclusive artifacts, no retries or score edits."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO=Path('/workspace/behavior-1k-solution')
HERE=REPO/'evaluations/pi05-15tasks'
OUT=Path(os.environ.get('B1K_EVAL_OUTPUT', '/workspace/evaluation-results/pi05-15tasks-40000'))
CKPT=Path('/workspace/checkpoints/pi05-15tasks/checkpoint_40000')
TASK_IDS=[5,7,11,13,14,15,16,17,18,21,24,25,29,30,45]
TABLE=HERE/'audit/datasets/behavior-1k/2026-challenge-demos/meta/tasks.jsonl'
TASK_NAMES={r['task_index']:r['task_name'] for r in map(json.loads,TABLE.read_text().splitlines())}
ENV={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMNIGIBSON_HEADLESS':'1',
     'OMNIGIBSON_DATA_PATH':'/workspace/BEHAVIOR-2026/datasets','PYTHONHASHSEED':'0',
     'XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_PYTHON_CLIENT_ALLOCATOR':'platform',
     'XLA_PYTHON_CLIENT_MEM_FRACTION':'0.5','TOKENIZERS_PARALLELISM':'false'}
children=[]

def status(event,**values):
    record={'time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':event,**values}
    print(json.dumps(record),flush=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')

def verify_result(directory,task,index):
    files=list((directory/'json').glob('*.json'))
    if len(files)!=1:raise RuntimeError(f'Expected one JSON in {directory}, found {len(files)}')
    r=json.loads(files[0].read_text())
    if r['task']!=TASK_NAMES[task] or r['instance_id']!=301+index or r['rollout_id']!=0:
        raise RuntimeError(f'Result identity mismatch: {files[0]}')
    if r['steps']<=0:raise RuntimeError('Empty rollout')
    videos=list((directory/'videos').glob('*.mp4'))
    if len(videos)!=1 or videos[0].stat().st_size==0:raise RuntimeError(f'Missing video in {directory}')
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',str(videos[0])],capture_output=True,text=True,check=True)
    streams=json.loads(probe.stdout)['streams']
    if not streams or int(streams[0].get('nb_frames',0))<1:raise RuntimeError('Video contains no frames')
    return r,str(files[0]),str(videos[0])

def exclusive_run(task,index,directory,validation=False):
    directory.mkdir(parents=True,exist_ok=False)
    cmd=['/workspace/behavior2026-env/bin/python','-m','omnigibson.eval.eval',
         '--task-name',TASK_NAMES[task],'--mode','public_test','--instance-indices',str(index),
         '--num-envs','1','--num-rollouts','1','--host','127.0.0.1','--port','8000',
         '--env-wrapper','omnigibson.eval.wrappers.DefaultWrapper','--output-dir',str(directory),'--write-video']
    if validation:cmd+=['--max-steps','300']
    (directory/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    status('rollout_start',task=task,index=index,validation=validation,output=str(directory))
    with (directory/'evaluator.log').open('x') as log:
        child=subprocess.Popen(cmd,cwd=REPO,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        children.append(child)
        code=child.wait()
        children.remove(child)
    if code:raise RuntimeError(f'Evaluator exit {code}: {directory}/evaluator.log')
    result,j,v=verify_result(directory,task,index)
    status('rollout_complete',task=task,index=index,validation=validation,json=j,video=v,
           steps=result['steps'],success=result['success'],q_score=result.get('q_score'))
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Refuse a second run of this launch; completed and incomplete artifacts are retained.
    (OUT/'launch.json').open('x').write(json.dumps({'pid':os.getpid(),'checkpoint':str(CKPT),'historical_training_config':'UNRECOVERABLE'})+'\n')
    ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    forbidden=('serve_checkpoint.py','omnigibson.eval.eval','run_instances.py')
    for line in ps.splitlines():
        if any(x in line for x in forbidden) and ('/python ' in line or '/python3 ' in line):
            raise RuntimeError(f'Existing policy/evaluator process: {line}')
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',8000))==0:raise RuntimeError('Port 8000 already occupied')
    if not (CKPT/'params/manifest.ocdbt').exists():raise RuntimeError('Checkpoint download incomplete')
    status('starting_shared_policy',checkpoint=str(CKPT))
    with (OUT/'logs/policy.log').open('x') as log:
        server=subprocess.Popen(['/workspace/policy-env/bin/python',str(HERE/'serve_checkpoint.py'),
             '--checkpoint',str(CKPT),'--audit-log',str(OUT/'logs/policy-audit.jsonl')],cwd=REPO,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        children.append(server)
        for _ in range(240):
            if server.poll() is not None:raise RuntimeError('Policy server exited during startup')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=2) as response:
                    if response.status==200:break
            except OSError:pass
            time.sleep(5)
        else:raise RuntimeError('Policy readiness timeout')
        status('policy_ready',pid=server.pid)
        validation=OUT/'validation/task-5-public-index-10'
        exclusive_run(5,10,validation,validation=True)
        events=[json.loads(s) for s in (OUT/'logs/policy-audit.jsonl').read_text().splitlines()]
        if not any(e['event']=='model_input' and e['task_id']==e['embedding_row']==5 for e in events):
            raise RuntimeError('Validation has no verified task-5 model input')
        if not any(e['event']=='finite_actions' and e['task_id']==5 and e['count']>=100 for e in events):
            raise RuntimeError('Validation has insufficient finite-action evidence')
        status('validation_artifacts_ready_for_review',output=str(validation))
        # Agent reviews the preserved validation artifacts before releasing scored evaluations.
        while not (OUT/'validation-reviewed.json').exists():
            if server.poll() is not None:raise RuntimeError('Policy server exited before validation review')
            time.sleep(5)
        status('validation_reviewed')
        for task in TASK_IDS:
            for index in range(10):
                expected=f'{TASK_NAMES[task]}_{301+index}_0.json'
                existing=list(OUT.rglob(expected))
                if existing:
                    status('skip_existing_result',task=task,index=index,paths=[str(p) for p in existing])
                    continue
                if server.poll() is not None:raise RuntimeError('Shared policy server exited')
                exclusive_run(task,index,OUT/f'task-{task}-{TASK_NAMES[task]}'/f'instance-{index}')
            status('task_complete',task=task)
        status('all_tasks_complete')

def stop(signum,frame):
    raise KeyboardInterrupt(f'Received signal {signum}')

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop)
    try:main()
    except BaseException as e:
        status('stopped',reason=repr(e))
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()

import hashlib,json,pathlib,requests,subprocess,os
root=pathlib.Path('/workspace/evaluation-results/ckpt3-10k-tasks-31-27')
repo=pathlib.Path('/workspace/behavior-1k-solution')
dest=repo/'evaluations/ckpt3-10k-tasks-31-27'
token=pathlib.Path('/root/.config/behavior-eval/github-token').read_text().strip()
s=requests.Session();s.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
api='https://api.github.com/repos/estjtmi/behavior-1k-solution'
tag='eval-ckpt3-10k-tasks-31-27-20261005'
r=s.get(api+'/releases/tags/'+tag); 
if r.status_code==404:
 r=s.post(api+'/releases',json={'tag_name':tag,'target_commitish':'eval/ckpt3-10k-tasks-31-27-20261005','name':'2026 evaluation: checkpoint 3, tasks 31 and 27','body':'Rollout videos for completed evaluations. Per-episode JSON metrics and the video index are on branch eval/ckpt3-10k-tasks-31-27-20261005. This run is in progress; assets are added as episodes complete.'})
r.raise_for_status(); release=r.json()
assets=s.get(api+'/releases/'+str(release['id'])+'/assets',params={'per_page':100});assets.raise_for_status()
existing={x['name']:x for x in assets.json()}; index=[]
for metric in sorted(root.glob('*/instance-*/json/*.json')):
 video=metric.parent.parent/'videos'/(metric.stem+'.mp4')
 if not video.exists():raise RuntimeError('Missing video: '+str(video))
 # Validate finalized container before publishing (ignore in-progress episodes).
 subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1',str(video)],check=True,stdout=subprocess.DEVNULL)
 digest=hashlib.sha256(video.read_bytes()).hexdigest()
 asset=existing.get(video.name)
 if asset is None:
  print('Uploading',video.name,video.stat().st_size,flush=True)
  with video.open('rb') as f:
   response=s.post(release['upload_url'].split('{')[0],params={'name':video.name},data=f,headers={'Content-Type':'video/mp4'},timeout=600)
  response.raise_for_status();asset=response.json()
 elif asset['size']!=video.stat().st_size:raise RuntimeError('Remote size mismatch: '+video.name)
 index.append({'task':metric.parent.parent.parent.name,'json':str(pathlib.Path('results')/metric.relative_to(root)),'video':asset['browser_download_url'],'bytes':video.stat().st_size,'sha256':digest})
(dest/'videos.json').write_text(json.dumps(index,indent=2)+'\n')
(dest/'ROLLOUT_VIDEOS.md').write_text('# Rollout videos\n\nFinalized videos are stored as [GitHub release assets]('+release['html_url']+').\n\n'+ '\n'.join('- ['+pathlib.Path(x['json']).stem+']('+x['video']+') — [JSON]('+x['json']+')' for x in index)+'\n')
subprocess.run(['git','add',str(dest/'videos.json'),str(dest/'ROLLOUT_VIDEOS.md')],cwd=repo,check=True)
subprocess.run(['git','-c','user.name=estjtmi','-c','user.email=estjtmi@users.noreply.github.com','commit','-m','Link completed rollout videos and JSON results'],cwd=repo)
env=dict(os.environ,GIT_ASKPASS='/root/.config/behavior-eval/git-askpass',GIT_TERMINAL_PROMPT='0')
subprocess.run(['git','push'],cwd=repo,env=env,check=True)
print('Published videos:',len(index),release['html_url'],flush=True)

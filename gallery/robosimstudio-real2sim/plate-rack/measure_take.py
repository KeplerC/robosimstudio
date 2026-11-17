"""Resumable all-frame measurement with check.py's original surface tests."""
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')})
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace

def work(task):
    folder,k,chunks,workers=task
    os.sched_setaffinity(0,{k%workers})
    import check
    path=Path(folder)/'check_chunks'/f'{k:03}.json'
    if not path.exists():
        result=check.measure(SimpleNamespace(take=folder,samples=40000,stride=1),k,chunks)
        tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(result));tmp.replace(path)
    return k

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('take');ap.add_argument('--jobs',type=int,default=8);ap.add_argument('--chunks',type=int,default=80)
    a=ap.parse_args();folder=Path(a.take);out=folder/'check_chunks';out.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for f in as_completed([pool.submit(work,(str(folder),k,a.chunks,a.jobs)) for k in range(a.chunks)]):print('checked batch',f.result(),flush=True)
    import check
    r=check.merge([json.loads((out/f'{k:03}.json').read_text()) for k in range(a.chunks)])
    for v in r['worst_penetration'].values():
        if v['depth_mm']<=-1e8:v.update(depth_mm=None,note='never within 1 mm of a plate')
    r['note']='Original check.py method; every 30 Hz recorded state; 40,000 samples plus vertices per plate and 80,000 plus vertices per fingertip. Resumable interleaved batches.'
    (folder/'penetration.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
    bad=any(v['depth_mm'] is not None and v['depth_mm']>0 for v in r['worst_penetration'].values()) or r['plates_closest_approach']['clearance_mm']<0
    raise SystemExit(int(bad))

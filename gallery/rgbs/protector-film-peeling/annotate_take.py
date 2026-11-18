"""Add measured per-frame bond telemetry to an unmodified path-traced take."""
from pathlib import Path
import json,sys,subprocess
import cv2,numpy as np
root=Path(__file__).resolve().parent / "run"
root.mkdir(parents=True, exist_ok=True)
name=sys.argv[1];meta=json.loads((root/f'{name}.json').read_text());metrics=json.loads((root/f'{name}_metrics.json').read_text())
video=root/f'{name}.mp4';raw=root/f'{name}_raw.mp4'
if not raw.exists():video.rename(raw)
samples=metrics['samples'];ts=np.array([r['time'] for r in samples]);cap=cv2.VideoCapture(str(raw));w=meta['width'];h=meta['height'];fps=meta['fps']
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{w}x{h}','-framerate',str(fps),'-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
rows=[]
for f in meta['frames']:
 ok,im=cap.read()
 if not ok:break
 t=f['t'];k=np.searchsorted(ts,t+1e-6,side='right')-1;r=samples[max(0,k)];peeled=0 if k<0 else r['peeled']
 scale=w/1920;pad=int(18*scale);height=int(108*scale)
 layer=im.copy();cv2.rectangle(layer,(pad,pad),(int(660*scale),pad+height),(22,29,33),-1);im=cv2.addWeighted(layer,.88,im,.12,0)
 color=(145,235,170) if peeled>=.95 else (240,240,240)
 cv2.putText(im,f'FILM PEELED  {peeled*100:5.1f}%',(pad+int(18*scale),pad+int(43*scale)),cv2.FONT_HERSHEY_SIMPLEX,.95*scale,color,2,cv2.LINE_AA)
 cv2.putText(im,f'{t:4.1f} s   |   target 95%   |   measured bonds',(pad+int(18*scale),pad+int(83*scale)),cv2.FONT_HERSHEY_SIMPLEX,.65*scale,(215,220,225),1,cv2.LINE_AA)
 proc.stdin.write(im.tobytes());rows.append(dict(frame=f['i'],time=t,peeled=peeled))
 if abs(t-36)<.01 or (peeled>=.95 and (len(rows)==1 or rows[-2]['peeled']<.95)):
  cv2.imwrite(str(root/f'{name}_reading_{t:05.2f}.png'),im)
proc.stdin.close();assert proc.wait()==0;cap.release();(root/f'{name}_readings.json').write_text(json.dumps(rows,indent=2))
print(name,'frames',len(rows),'first >=95%',next((r for r in rows if r['peeled']>=.95),None))

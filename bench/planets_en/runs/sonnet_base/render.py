import numpy as np, subprocess
from PIL import Image
A='input/art/'
def L(p,w=None,h=None):
    im=Image.open(A+p).convert('RGBA')
    if w or h:
        if h: w=round(im.width*h/im.height)
        else: h=round(im.height*w/im.width)
        im=im.resize((w,h),Image.LANCZOS)
    return im
def arr(im):
    a=np.asarray(im).astype(np.float32); a[...,:3]*=a[...,3:]/255; return a  # premult
bg=np.asarray(Image.open(A+'background.png').convert('RGB')).astype(np.float32)
def put(frame,im,x,y,op):
    if op<=0.001: return
    x=int(round(x));y=int(round(y));h,w=im.shape[:2]
    x0,y0=max(x,0),max(y,0);x1,y1=min(x+w,1920),min(y+h,1080)
    if x0>=x1 or y0>=y1: return
    s=im[y0-y:y1-y,x0-x:x1-x]; a=s[...,3:]/255*op
    frame[y0:y1,x0:x1]=frame[y0:y1,x0:x1]*(1-a)+s[...,:3]*op
lineup=arr(L('lineup.png')); l8=arr(L('labels/eight_planets.png'))
sun=arr(L('sun.png',h=1000))
names=['mercury','venus','earth','mars','jupiter','saturn','uranus','neptune']
starts=[6.18,10.8,16.24,18.48,20.98,26.24,31.3,33.7]
kw=[8.0,11.35,17.5,19.9,23.1,27.95,32.8,36.3]
P=[]
for n in names:
    im=L(f'planets/{n}.png',w=1100) if n=='saturn' else L(f'planets/{n}.png',h=600)
    P.append((arr(im),arr(L(f'labels/{n}.png'))))
D=39.29; N=round(D*60)
def ease(t): t=min(max(t,0),1); return 1-(1-t)**3
def seg(t,a,d): return ease((t-a)/d)
pr=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','60','-i','-','-i','input/narration.wav',
 '-c:v','libx264','-pix_fmt','yuv420p','-crf','16','-preset','medium','-r','60','-c:a','aac','-b:a','192k','-t',str(D),'out/final.mp4'],stdin=subprocess.PIPE)
for i in range(N):
    t=i/60; f=bg.copy()
    # lineup
    lo=seg(t,0.4,0.9)*(1-seg(t,starts[0]-0.1,0.5))
    if lo>0:
        put(f,lineup,60,(1080-420)//2+40+(1-lo)*30,lo)
        put(f,l8,(1920-681)//2,300-(1-lo)*20-100+0,lo)
    # sun
    so=seg(t,starts[0]-0.1,0.7)*(1-seg(t,starts[1]-0.1,0.5))
    put(f,sun,-720-(1-so)*120,40,so)
    for k,(p,lb) in enumerate(P):
        a=starts[k]; b=starts[k+1] if k<7 else 99
        if t<a-0.05 or t>b+0.6: continue
        o=seg(t,a,0.6)*(1-seg(t,b-0.1,0.5)) if k<7 else seg(t,a,0.6)
        ph,pw=p.shape[:2]
        cx=620+(1-seg(t,a,0.8))*260-(seg(t,b-0.1,0.6)*200 if k<7 else 0)
        put(f,p,cx-pw/2,540-ph/2,o)
        lh,lw=lb.shape[:2]
        lo2=seg(t,kw[k]-0.5,0.5)*(1-seg(t,b-0.1,0.4)) if k<7 else seg(t,kw[k]-0.5,0.5)
        lx=620+pw/2+80+(1-seg(t,kw[k]-0.5,0.7))*60
        put(f,lb,lx,540-lh/2,lo2)
    pr.stdin.write(np.clip(f,0,255).astype(np.uint8).tobytes())
pr.stdin.close(); pr.wait()

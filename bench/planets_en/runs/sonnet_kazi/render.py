import numpy as np, subprocess
from PIL import Image
A='input/art/'
ld=lambda p:Image.open(A+p).convert('RGBA')
bg=ld('background.png').convert('RGB')
FPS=60; DUR=39.29; N=round(DUR*FPS)
names=['mercury','venus','earth','mars','jupiter','saturn','uranus','neptune']
kw=[8.08,11.40,17.58,19.98,23.20,28.02,32.88,36.38]
HIT=2.92
pl={n:ld(f'planets/{n}.png') for n in names}
lb={n:ld(f'labels/{n}.png') for n in names}
lab8=ld('labels/eight_planets.png'); lineup=ld('lineup.png'); sun=ld('sun.png')
def ease(t): t=min(max(t,0),1); return 1-(1-t)**3
def paste(canvas,im,cx,cy,scale=1.0,alpha=1.0):
    if alpha<=0: return
    w,h=max(1,int(im.width*scale)),max(1,int(im.height*scale))
    r=im.resize((w,h),Image.LANCZOS)
    if alpha<1:
        a=r.getchannel('A').point(lambda v:int(v*alpha)); r.putalpha(a)
    canvas.alpha_composite(r,(0,0)) if False else None
    x,y=int(round(cx-w/2)),int(round(cy-h/2))
    canvas.paste(r,(x,y),r)
def frame(i):
    t=i/FPS
    c=bg.copy()
    # lineup + label
    ls=kw[0]-0.8; le=kw[0]-0.5
    if t<le:
        tin=ease((t-(HIT-0.05-0.45))/0.45) if t<HIT+1 else 1
        tout=ease((t-ls)/(le-ls)) if t>ls else 0
        a=tin*(1-tout)
        d=(1-ease((t-(HIT-0.5))/0.45))*0
        paste(c,lineup,960,590+ (1-tin)*30,1.0+0.01*(t-2.5),a)
        if t>HIT-0.3:
            la=ease((t-(HIT-0.3))/0.25)*(1-tout)
            paste(c,lab8,960,200+(1-la)*-20,1.0,la)
    # sun during mercury line
    ss=6.0; se=kw[1]-0.8
    if ss<t<se+0.4:
        sa=ease((t-ss)/0.6)*(1-ease((t-se)/0.4))
        paste(c,sun,-330-(1-ease((t-ss)/0.9))*200,540,1000/1440*1.0,sa)
    for k,n in enumerate(names):
        h=kw[k]-0.05; s=h-0.45
        e=(kw[k+1]-0.8) if k<7 else 99
        if t<s or t>e+0.35: continue
        pin=ease((t-s)/0.45); pout=ease((t-e)/0.3) if t>e else 0
        a=pin*(1-pout)
        im=pl[n]
        sc=(1100/im.width) if n=='saturn' else (600/im.height)
        sc*=1+0.04*(t-s)/3.0*0+0.012*np.sin((t-s)*1.2)+ (1-pin)*0.0
        cx=650+(1-pin)*500-pout*300; cy=540+4*np.sin((t-s)*1.5)
        paste(c,im,cx,cy,sc,a)
        # label
        lt=t-(h-0.25)
        if lt>0:
            la=ease(lt/0.25)*(1-pout)
            hw=im.width*sc/2
            L=lb[n]
            paste(c,L,650+hw+60+L.width/2-(1-ease(lt/0.25))*50-pout*300,540,1.0,la)
    return c.convert('RGB')
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','60','-i','-',
 '-i','input/narration.wav','-map','0:v','-map','1:a','-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p',
 '-c:a','aac','-b:a','320k','-ar','48000','-t',str(DUR),'out/final.mp4'],stdin=subprocess.PIPE)
for i in range(N): p.stdin.write(frame(i).tobytes())
p.stdin.close(); p.wait()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

EXTS={'.png','.jpg','.jpeg','.webp'}

def tiles(root,w,h):
    out=[]
    for p in root.rglob('*'):
        if p.suffix.lower() not in EXTS: continue
        try:
            if Image.open(p).size==(w,h): out.append(p)
        except: pass
    return sorted(out)

def lum(x):
    x=np.asarray(x,float)/255
    return float(.2126*x[0]+.7152*x[1]+.0722*x[2])

def color_clusters(arr,k=5):
    lab=cv2.cvtColor(arr.astype(np.uint8),cv2.COLOR_RGB2LAB)
    data=lab.reshape(-1,3).astype(np.float32)
    crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.4)
    _,labels,centers=cv2.kmeans(data,k,None,crit,5,cv2.KMEANS_PP_CENTERS)
    labels=labels.reshape(arr.shape[:2])
    centers_rgb=cv2.cvtColor(centers.reshape(1,k,3).astype(np.uint8),cv2.COLOR_LAB2RGB).reshape(k,3)
    return labels, centers_rgb

def analyze(path):
    arr=np.asarray(Image.open(path).convert('RGB'))
    h,w,_=arr.shape
    labels, centers=color_clusters(arr,5)
    b=max(12,w//10)
    corner=np.concatenate([labels[:b,:b].ravel(),labels[:b,-b:].ravel(),labels[-b:,:b].ravel(),labels[-b:,-b:].ravel()])
    bg_idx=Counter(corner.tolist()).most_common(1)[0][0]
    bg=centers[bg_idx]

    yy,xx=np.indices((h,w))
    center_prior=((xx-w/2)/(w/2))**2+((yy-h/2)/(h/2))**2<.92
    mask=np.zeros((h,w),np.uint8)
    for idx in range(len(centers)):
        if idx==bg_idx: continue
        cm=(labels==idx).astype(np.uint8)
        # score clusters by center occupancy and color distance from bg
        occ_center=cm[center_prior].mean(); occ_corner=cm[~center_prior].mean()
        dist=np.linalg.norm(centers[idx].astype(float)-bg.astype(float))
        if dist>16 and occ_center>occ_corner*1.15:
            mask|=cm
    mask=(mask*255).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

    # retain meaningful central components
    n,lab,stats,cents=cv2.connectedComponentsWithStats(mask,8)
    clean=np.zeros_like(mask)
    comps=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        cx,cy=cents[i]
        if area<35: continue
        if not (w*.08<cx<w*.92 and h*.05<cy<h*.95): continue
        if bw>2 and bh>5:
            clean[lab==i]=255; comps.append((x,y,bw,bh,area))
    mask=clean
    ys,xs=np.where(mask>0)
    bbox=None
    if len(xs):
        bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        x1,y1,x2,y2=bbox
        fgpx=arr[mask>0]
        fg=np.median(fgpx,axis=0)
        bw=x2-x1; bh=y2-y1; cx=(x1+x2)/2/w; cy=(y1+y2)/2/h
    else:
        fg=bg; bw=bh=0; cx=cy=.5

    # underline detection from long thin connected components
    underline='none'; line_strength=0.0
    if bbox:
        x1,y1,x2,y2=bbox
        candidates=[]
        for x,y,cw,ch,area in comps:
            aspect=cw/max(ch,1)
            if aspect>4 and cw>0.35*w and ch<0.12*h:
                candidates.append((x,y,cw,ch,area))
        if candidates:
            line=max(candidates,key=lambda z:z[2])
            ly=line[1]+line[3]/2
            rel=(ly-y1)/max(1,bh)
            underline='below' if rel>.78 else 'overlap'
            line_strength=line[2]/w

    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    smooth=cv2.GaussianBlur(gray,(0,0),3)
    resid=gray.astype(float)-smooth.astype(float)
    corners=np.concatenate([resid[:40,:40].ravel(),resid[:40,-40:].ravel(),resid[-40:,:40].ravel(),resid[-40:,-40:].ravel()])
    gx=np.abs(np.diff(gray.astype(float),axis=1)).mean(); gy=np.abs(np.diff(gray.astype(float),axis=0)).mean()

    return {
      'file':path.name,'background_rgb':[int(x) for x in bg],'foreground_rgb':[int(round(x)) for x in fg],
      'contrast_luma':round(abs(lum(fg)-lum(bg)),4),'bbox':bbox,
      'bbox_width_ratio':round(bw/w,4),'bbox_height_ratio':round(bh/h,4),
      'center_x_ratio':round(cx,4),'center_y_ratio':round(cy,4),
      'texture_std':round(float(np.std(corners)),4),'edge_strength':round(float((gx+gy)/2),4),
      'underline_mode':underline,'line_strength':round(line_strength,4)
    }, mask

def quant(v):
    a=np.asarray(v,float); ps=(.05,.25,.5,.75,.95)
    return {str(p):round(float(np.quantile(a,p)),4) for p in ps}

def palette(vals,step=16):
    c=Counter(tuple(min(255,max(0,int(round(x/step)*step))) for x in v) for v in vals)
    return [{'rgb':list(k),'count':n} for k,n in c.most_common(24)]

def summary(items):
    return {'tile_count':len(items),'tile_size':[200,200],
      'background_palette':palette([x['background_rgb'] for x in items]),
      'foreground_palette':palette([x['foreground_rgb'] for x in items]),
      'bbox_width_ratio':quant([x['bbox_width_ratio'] for x in items if x['bbox']]),
      'bbox_height_ratio':quant([x['bbox_height_ratio'] for x in items if x['bbox']]),
      'center_x_ratio':quant([x['center_x_ratio'] for x in items if x['bbox']]),
      'center_y_ratio':quant([x['center_y_ratio'] for x in items if x['bbox']]),
      'contrast_luma':quant([x['contrast_luma'] for x in items]),
      'texture_std':quant([x['texture_std'] for x in items]),
      'edge_strength':quant([x['edge_strength'] for x in items]),
      'underline_modes':dict(Counter(x['underline_mode'] for x in items))}

def report(records,out):
    cols=4; cellw=200; cellh=250; rows=math.ceil(len(records)/cols)
    can=Image.new('RGB',(cols*cellw,rows*cellh),'white'); d=ImageDraw.Draw(can)
    for i,(item,path,mask) in enumerate(records):
        x=(i%cols)*cellw; y=(i//cols)*cellh
        im=Image.open(path).convert('RGB'); can.paste(im,(x,y))
        if item['bbox']:
            b=item['bbox']; d.rectangle((x+b[0],y+b[1],x+b[2],y+b[3]),outline='red',width=2)
        d.text((x+3,y+202),f"{item['file'][:21]}\nbox {item['bbox_width_ratio']:.2f}x{item['bbox_height_ratio']:.2f} {item['underline_mode']}",fill='black')
    can.save(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('--output',type=Path,default=Path('captcha_style.json')); ap.add_argument('--report',type=Path,default=Path('captcha_style_report.png')); ap.add_argument('--width',type=int,default=200); ap.add_argument('--height',type=int,default=200)
    a=ap.parse_args(); ps=tiles(a.input,a.width,a.height)
    if not ps: raise SystemExit('No 200x200 tiles found')
    rec=[]
    for p in ps:
        item,mask=analyze(p); rec.append((item,p,mask))
    items=[r[0] for r in rec]
    a.output.write_text(json.dumps({'summary':summary(items),'tiles':items},indent=2),encoding='utf-8')
    report(rec,a.report)
    print(f'Analyzed {len(items)} tiles\nCalibration: {a.output}\nReport: {a.report}')
if __name__=='__main__': main()

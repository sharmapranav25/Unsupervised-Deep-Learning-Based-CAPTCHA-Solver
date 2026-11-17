"""Pixel continuity evidence for long straight lines across several blocks."""
import cv2
import numpy as np

def supported_lines(image):
    gray=np.asarray(image.convert('L'));h,w=gray.shape
    edges=cv2.Canny(gray,40,100)
    detected=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(10,w//16),minLineLength=max(12,round(.3*w)),maxLineGap=6)
    accepted=[]
    if detected is None:return accepted
    for x0,y0,x1,y1 in detected[:,0]:
        dx,dy=float(x1-x0),float(y1-y0);length=float(np.hypot(dx,dy));nx,ny=-dy/length,dx/length
        t=np.linspace(0,1,max(2,round(length)));x=x0+t*dx;y=y0+t*dy
        def sample(offset):
            xx=np.clip(np.rint(x+offset*nx).astype(int),0,w-1);yy=np.clip(np.rint(y+offset*ny).astype(int),0,h-1)
            return gray[yy,xx].astype(float)
        # Canny follows a boundary: look within one pixel for its stroke.
        center=np.stack([sample(k) for k in (-1,0,1)])
        backdrop=(sample(-4)+sample(4))/2
        contrast=np.max(np.abs(center-backdrop),axis=0)
        support=float(np.mean(contrast>=20))
        if support<.8:continue
        midpoint=np.array([(x0+x1)/2,(y0+y1)/2]);angle=float(np.degrees(np.arctan2(dy,dx))%180)
        if any(min(abs(angle-r['angle']),180-abs(angle-r['angle']))<5 and np.linalg.norm(midpoint-np.array(r['midpoint']))<6 for r in accepted):continue
        accepted.append(dict(points=[int(x0),int(y0),int(x1),int(y1)],length=length,angle=angle,support=support,midpoint=midpoint.tolist()))
    return accepted

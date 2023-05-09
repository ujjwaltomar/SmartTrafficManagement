import cv2
import torch
import numpy as np
from tracker import *


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('tvid.mp4')

count=0
tracker = Tracker()
counter=0
area_1=set()

b=model.names[2] = 'car'

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)
id=-1
area1=[(225,380),(450,395),(460,370),(260,360)]
while True:
    
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,600))
    cv2.line(frame,(75,420),(1000,420),(255,225,255),1)
    results=model(frame)
    list=[]
    a=results.pandas().xyxy[0]
    

    for index,rows in results.pandas().xyxy[0].iterrows():
      
      x=int(rows[0])
      y=int(rows[1])
      x1=int(rows[2])
      y1=int(rows[3])
      cl=int(rows[5])
      b=str(rows['name'])
      list.append([x,y,x1,y1])
      #print(rows)
        
      idx_bbox=tracker.update(list)
      #print(idx_bbox)
      for bbox in idx_bbox:
        id=bbox[4]
      if cl==2:
          
          #,id=bbox
          cv2.rectangle(frame,(x,y),(x1,y1),(0,0,225),2)
          rectx1,recty1=((x+x1)/2,(y+y1)/2)
          reccenter=int(rectx1),int(recty1)
          cx=reccenter[0]
          cy=reccenter[1]
          cv2.circle(frame,(cx,cy),3,(0,225,0),-1)
          cv2.putText(frame,str(b),(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),1)
          #cv2.circle(frame,(x1,y1),4,(0,255,0),-1)
          #result=cv2.pointPolygonTest(np.array(area1,np.int32),((x3, y3)),False)
          #if result > 0 :
          #    area_1.add(id)
          if cy<=(420+10) and cy>=(420-10) and cx<=600:
              counter+=1
              area_1.add(id)
              cv2.line(frame,(75,420),(600,420),(0,225,0),1)
          
          if cy<=(420+10) and cy>=(420-10) and cx>600:
              counter+=1
              area_1.add(id)
              cv2.line(frame,(601,420),(1000,420),(0,225,0),1)
          
          
    
    #cv2.polylines(frame,[np.array(area1,np.int32)],True,(225,0,255),1)
    #print(x2,y2,x3,y3)
    #a1=len(area_1)
    #print(a1)
    cv2.putText(frame,str(counter),(100,420),cv2.FONT_HERSHEY_PLAIN,3,(0,225,0),3)
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


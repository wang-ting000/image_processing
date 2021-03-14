import numpy as np
import cv2

drawing = False
ix,iy = -1,-1

def empty():
    pass

#create a window:
img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('brushing')

#create trackbars
cv2.createTrackbar('r','brushing',0,255,empty)
cv2.createTrackbar('g','brushing',0,255,empty)
cv2.createTrackbar('b','brushing',0,255,empty)
cv2.createTrackbar('brush size','brushing',0,25,empty)



def draw(event,x,y,flags,param):
    global drawing,ix,iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x,y = ix,iy
    elif event == cv2.EVENT_MOUSEMOVE:
        drawing = True
        cv2.circle(img,(x,y),size,(b,g,r),0)

while True:
    cv2.imshow('paint',img)
    if cv2.waitKey(1)==27:
        break
    else:

        # get position
        r = cv2.getTrackbarPos('r', 'brushing')
        g = cv2.getTrackbarPos('g', 'brushing')
        b = cv2.getTrackbarPos('b', 'brushing')
        size = cv2.getTrackbarPos('brush size', 'brushing')
        cv2.setMouseCallback('paint', draw)


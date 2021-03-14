# image_processing
digital image processing

1. 阈值处理  

```retval,otsu = cv2.threshold(grayscales,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)```

2. 读取摄像头
            
       cap = cv2.VideoCapture(0)
            #打开电脑的摄像头
        cap.set(10,100)

        while True:
            success,img = cap.read()
            cv2.imshow('video',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #按q退出
                break
                
3. 灰度函数
          
        imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
or
        
        img = cv2.imread(filaname,cv2.IMREAD_GRAYSCALE)
        
4. 高斯模糊
    
        imgBlur = cv2.GaussianBlur(imgGrey,(7,7),0)
        ##kernel需要为奇数
        
5. 边缘函数
        
        imgCanny = cv2.Canny(img,100,100)
        
or

边缘检测（微分）

            lapcian = cv2.Laplacian(img,cv2.CV_64F)
            
        
6. 扩张函数

        kernel = np.ones((5,5),np.uint8)
        
        imgDialation = cv2.dilate(imgCanny,kernel,iterations=5)

7. 腐蚀函数

        imgEroded = cv2.erode(imgDialation,kernel,iterations=1)
        
8. 画图

        cv2.putText,cv2.CIrcle...
        
9. 连接

        hor = np.hstack((img,img)) 
        ver = np.vstack((img,img))
        
10. 定义拼图的函数：
```
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
```
11. 检测颜色
```
cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars',640,240)
def empty(a):
    pass
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)


while True:
    img = cv2.imread('lambo.png')
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow('combined',imgResult)
    cv2.imshow('origin',img)
    cv2.imshow('HSV',imgHSV)
    cv2.imshow('Mask',mask)


    cv2.waitKey(1)
```
12. 鼠标事件：有**cv.EVENT_MOUSEMOVE     0   鼠标移动事件
        **cv.EVENT_LBUTTONDOWN   1   鼠标左键按下事件
        **cv.EVENT_RBUTTONDOWN   2   鼠标右键按下事件
        **cv.EVENT_MBUTTONDOWN   3   鼠标中键按下事件
        **cv.EVENT_LBUTTONUP     4   鼠标左键释放事件
        **cv.EVENT_RBUTTONUP     5   鼠标右键释放事件
        **cv.EVENT_MBUTTONUP     6   鼠标中键释放事件
        **cv.EVENT_LBUTTONBLCLK  7   鼠标左键双击事件
        **cv.EVENT_RBUTTONBLCLK  8   鼠标右键双击事件
        **cv.EVENT_MBUTTONBLCLK  9   鼠标中键双击事件
        **cv.EVENT_MOUSEWHEEL    10  滑动滚轮上下滚动
        **cv.EVENT_MOUSEHWHEEL   11  滑动滚轮左右滚动**

```
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv.circle(img,(x,y),5,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv.circle(img,(x,y),5,(0,0,255),-1)

img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
cv.destroyAllWindows()
```
# some practices for homework and learning

## tutorial 1

题目：**找出某坐标对应的棋盘格子并判断是白棋还是黑棋**

[matlab代码](board_position.m)  

[python代码](board_position.py)

题目：**找不同并数有几处**  

[matlab代码](Tut1.m)  

[python代码](find_diff.py)

[作业](一、图像的基本操作.pdf)

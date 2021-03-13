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
# some practices for homework and learning

## tutorial 1

题目：**找出某坐标对应的棋盘格子并判断是白棋还是黑棋**

[matlab代码](board_position.m)  

[python代码](board_position.py)

题目：**找不同并数有几处**  

[matlab代码](Tut1.m)  

[python代码](find_diff.py)

[作业](一、图像的基本操作.pdf)

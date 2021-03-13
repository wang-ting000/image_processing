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
        
#**some practices for homework and learning**

# tutorial 1

题目：**找出某坐标对应的棋盘格子并判断是白棋还是黑棋**

[matlab代码](board_position.m)  

[python代码](board_position.py)

题目：**找不同并数有几处**  

[matlab代码](Tut1.m)  

[python代码](find_diff.py)

[作业](一、图像的基本操作.pdf)

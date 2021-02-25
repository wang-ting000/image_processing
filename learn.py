import cv2
import numpy as np

##--load image--##
'''img = cv2.imread('fig.png')
cv2.imshow('little cat',img)
# the fig will disappear imediately
cv2.waitKey(0)
#'0'represent show forever'''

##--load camera--##
'''cap = cv2.VideoCapture(0)
#打开电脑的摄像头
cap.set(10,100)

while True:
    success,img = cap.read()
    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #按q退出
        break'''

##--functions--##
'''img = cv2.imread('fig.png')
imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#灰度函数
imgBlur = cv2.GaussianBlur(imgGrey,(7,7),0)
#模糊函数，ksize需要奇数
imgCanny = cv2.Canny(img,100,100)
#边缘函数
kernel = np.ones((5,5),np.uint8)
imgDialation = cv2.dilate(imgCanny,kernel,iterations=5)
#扩张
imgEroded = cv2.erode(imgDialation,kernel,iterations=1)
cv2.imshow('Gray',imgGrey)
cv2.imshow('Blur',imgBlur)
cv2.imshow('Canny',imgCanny)
cv2.imshow('Dilation',imgDialation)
cv2.imshow('Erodtion',imgEroded)
cv2.waitKey(0)'''

##--reshape--##
'''img = cv2.imread('fig.png')
print(img.shape)
imgResize = cv2.resize(img,(300,400))
# 只需要定义宽度和高度
imgCropped = img[600:800,200:500]
#[高度，宽度]
cv2.imshow('cat',imgCropped)
cv2.waitKey(0)
'''

##--draw shapes--##
'''img = np.zeros((512,512,3),np.uint8)
#像素
#img[200:300] = 55,13,22

cv2.line(img,(0,0),(img.shape[0],img.shape[1]),(0,255,0),3)
#画线
cv2.rectangle(img,(0,0),(200,350),(250,28,39),cv2.FILLED)
cv2.circle(img,(400,50),20,(255,255,0),5)
cv2.putText(img,'world',(300,300),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)

cv2.imshow('img',img)
cv2.waitKey(0)'''

##--spird perspective--##
'''img = cv2.imread('fig.png')
width,height = 250,350
pst1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
pst2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pst1,pst2)
imgOutput = cv2.warpPerspective(img,matrix,(width,height))


cv2.imshow('it',imgOutput)
cv2.waitKey(0)'''

##--connect--##
img = cv2.imread('fig.png')
hor = np.hstack((img,img))

cv2.imshow('hor',hor)
cv2.waitKey(0)
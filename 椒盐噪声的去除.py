# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:13:26 2021

@author: Wang
"""
from numpy import random
import cv2
import numpy as np

# 定义函数，第一个参数是缩放比例，第二个参数是需要显示的图片组成的元组或者列表
def ManyImgs(scale, imgarray):
    rows = len(imgarray)         # 元组或者列表的长度
    cols = len(imgarray[0])      # 如果imgarray是列表，返回列表里第一幅图像的通道数，如果是元组，返回元组里包含的第一个列表的长度
    # print("rows=", rows, "cols=", cols)

    # 判断imgarray[0]的类型是否是list
    # 是list，表明imgarray是一个元组，需要垂直显示
    rowsAvailable = isinstance(imgarray[0], list)

    # 第一张图片的宽高
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    # print("width=", width, "height=", height)

    # 如果传入的是一个元组
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # 遍历元组，如果是第一幅图像，不做变换
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                # 将其他矩阵变换为与第一幅图像相同大小，缩放比例为scale
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)
                # 如果图像是灰度图，将其转换成彩色显示
                if  len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        # 创建一个空白画布，与第一张图片大小相同
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows   # 与第一张图片大小相同，与元组包含列表数相同的水平空白图像
        for x in range(0, rows):
            # 将元组里第x个列表水平排列
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)   # 将不同列表垂直拼接
    # 如果传入的是一个列表
    else:
        # 变换操作，与前面相同
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # 将列表水平排列
        hor = np.hstack(imgarray)
        ver = hor
    return ver
# 创建椒盐噪声函数
def salt_and_pepper_noise(img,rou):
    '''
    添加椒盐噪声
    img:原始图片
    rou:噪声比例
    '''
    img = cv2.imread(img)
    img_noised = np.zeros(np.shape(img),np.uint8)
    noise_out = np.zeros(np.shape(img),np.uint8)
    thres = 1 - rou
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            num = random.random()#随机生成0-1之间的数字
            if num < rou:#如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                img_noised[i][j] = 0
                noise_out[i][j] = 0
            elif num > thres:#如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                img_noised[i][j] = 255
                noise_out[i][j] = 255
            else:
                img_noised[i][j] = img[i][j]#其他情况像素点不变
                noise_out[i][j] = 100
    result = [noise_out,img_noised]#返回椒盐噪声和加噪图像
    return result
#生成椒盐噪声
sp_noise, img_noise1 = salt_and_pepper_noise('lena.png', 0.02)
img_noise1 = ManyImgs(0.5, ([img_noise1]))
sp_noise, img_noise2 = salt_and_pepper_noise('lena.png', 0.04)
img_noise2 = ManyImgs(0.5, ([img_noise2]))
sp_noise, img_noise3 = salt_and_pepper_noise('lena.png', 0.1)
img_noise3 = ManyImgs(0.5, ([img_noise3]))
sp_noise, img_noise4 = salt_and_pepper_noise('lena.png', 0.2)
img_noise4 = ManyImgs(0.5, ([img_noise4]))
noise = np.hstack((img_noise1,img_noise2,img_noise3,img_noise4))
'''#形态学变换（效果不好）
kernel = np.ones((5,5),np.uint8)
opening1 = cv2.morphologyEx(img_noise1, cv2.MORPH_OPEN, kernel)
opening2 = cv2.morphologyEx(img_noise1, cv2.MORPH_CLOSE, kernel)
opening3 = cv2.morphologyEx(img_noise1, cv2.MORPH_GRADIENT, kernel)
opening4 = cv2.morphologyEx(img_noise1, cv2.MORPH_CROSS, kernel)
opening = np.hstack((opening1,opening2,opening3,opening4))'''
##使用高斯模糊，效果一般
'''blur1 = cv2.GaussianBlur(img_noise1,(3,3),0)
blur2 = cv2.GaussianBlur(img_noise2,(3,3),0)
blur3 = cv2.GaussianBlur(img_noise3,(3,3),0)
blur4 = cv2.GaussianBlur(img_noise4,(3,3),0)
blur = np.hstack((blur1,blur2,blur3,blur4))
result = np.vstack((noise,blur))'''
##小波变换，基于此可以进行图像压缩
import pywt
cA,(cH,cV,cD) = pywt.dwt2(img_noise1,'haar')
cH = cH*0
cV = cV*0
#cD = cD*0
img_noise1 = pywt.idwt2((cA,(cH,cV,cD)),'haar')
cv2.imwrite('rimg.png',np.uint8(img_noise1))
wav1 = cv2.imread('rimg.png')
'----------------------------------------'
cA,(cH,cV,cD) = pywt.dwt2(img_noise4,'haar')
cH = cH*0
cV = cV*0
#cD = cD*0
img_noise4 = pywt.idwt2((cA,(cH,cV,cD)),'haar')
cv2.imwrite('rimg4.png',np.uint8(img_noise4))
wav4 = cv2.imread('rimg4.png')
'-----------------------------------------'
cA,(cH,cV,cD) = pywt.dwt2(img_noise2,'haar')
cH = cH*0
cV = cV*0
cD = cD*0
img_noise2 = pywt.idwt2((cA,(cH,cV,cD)),'haar')
cv2.imwrite('rimg2.png',np.uint8(img_noise2))
wav2 = cv2.imread('rimg2.png')
'----------------------------------------------'
cA,(cH,cV,cD) = pywt.dwt2(img_noise3,'haar')
cH = cH*0
cV = cV*0
cD = cD*0
img_noise3 = pywt.idwt2((cA,(cH,cV,cD)),'haar')
cv2.imwrite('rimg3.png',np.uint8(img_noise3))
wav3 = cv2.imread('rimg3.png')
cv2.imshow('noise',noise)
cv2.imshow('wav1',wav1)
cv2.imshow('wav2',wav2)
cv2.imshow('wav3',wav3)
cv2.imshow('wav4',wav4)
#cv2.imshow('re',result)
cv2.waitKey(0)




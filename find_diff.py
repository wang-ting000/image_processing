import cv2
import numpy as np
import pandas as pd

im = cv2.imread('dif.png')
im1=im[:,0:350,:]
im2=im[:,350:701,:]
'''im_diff = (np.asarray(im1)-np.asarray(im2)).tolist()
print(np.shape(im_diff))'''
im_diff=cv2.subtract(im1,im2)
im_diff = cv2.cvtColor(im_diff,cv2.COLOR_RGB2GRAY)
print(np.shape(im_diff))
im_diff=np.where(im_diff>40,1,0)
##
im_diff = np.asarray([im_diff*255,np.zeros_like(np.asarray(im_diff)),np.zeros_like(np.asarray(im_diff))])
print(np.shape(im_diff))
cv2.imshow('diff',im_diff)


cv2.waitKey(0)

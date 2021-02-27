import cv2
import numpy as np
import pandas as pd

im = cv2.imread('dif.png')

im1 = im[:, 0:350, :]
im2 = im[:, 350:701, :]

'''im_diff = (np.asarray(im1)-np.asarray(im2)).tolist()
print(np.shape(im_diff))'''
im_diff = cv2.subtract(im1, im2)
cv2.imshow('diff', im_diff)
im_diff = cv2.cvtColor(im_diff, cv2.COLOR_RGB2GRAY)


sub = im_diff.reshape(1, np.size(im_diff))
sub = np.where(sub > 40, 1, 0)
im_diff = sub.reshape(np.shape(im_diff))
im_diff_cw=255*np.array(im_diff, dtype='uint8')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,15), anchor=None)
im_diff_cw = cv2.dilate(im_diff_cw,kernel)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im_diff_cw, connectivity=8)
print(num_labels)
im_diff = im_diff.tolist()
a,b = np.shape(im_diff)


##
for i in range(a):
    for j in range(b):
        im_diff[i][j] = [im_diff[i][j]*255,0,0]  #opencv的颜色体系是BGR

print(np.shape(im_diff))
im_diff=np.array(im_diff, dtype='uint8')

im_diff = cv2.addWeighted(im_diff,4,im2,0.5,0)
im_diff = np.hstack((im1,im2,im_diff))



cv2.imshow('diff', im_diff)





cv2.waitKey(0)

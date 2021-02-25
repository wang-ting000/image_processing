import cv2
import numpy as np

im = cv2.imread('chess.png')


cv2.imshow('chess',im)
cv2.waitKey(0)
def board_position(im,row,col):
    [rows,columns,n] = np.shape(im)
    board_size = int(rows/8)
    row_st=row*board_size+1
    col_st=col*board_size+1
    row_end=row_st+board_size-1
    col_end=col_st+board_size-1
    print(row_st,row_end,col_st,col_end)

    im_crop=im[row_st:row_end,col_st:col_end]
    cv2.imshow('cropped',im_crop)

    im_gray =cv2.cvtColor(im_crop,cv2.COLOR_BGR2GRAY)

    n = np.size(im_gray)-np.count_nonzero(im_gray)
    m = np.size(im_gray)
    im_gray = [i-im_gray for i in im_gray ]
    if np.count_nonzero(im_gray) == np.size(im_gray):
        #cv2.addText(im_crop,'empty',(300,300),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)
        print('empty')
    else:
        white_pixel = int(n/m*100)
        if white_pixel>80:
            #cv2.addText(im_crop,'white',(300,300),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)
            print('white')
        else:
            #cv2.addText(im_crop,'black',(300,300),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)
            print('black')

    cv2.waitKey(0)

board_position(im,7,1)
%% Activity 2: Spot the Difference

im = imread('spot_the_difference.png');
im_info = imfinfo('spot_the_difference.png') %该函数用于获取一张图片的具体信息。这些具体信息包括图片的格式、尺寸、颜色数量、修改时间等等
im1 = im(:,1:350,:);  
im2 = im(:,351:700,:);  %两张图分别占据一半的宽度
im_diff=im1-im2;  %找差别
im_diff=rgb2gray(im_diff);
im_diff=im_diff>40; %True则为1，False为0


im_diff=cat(3,im_diff*255,zeros(size(im_diff)),zeros(size(im_diff))); %构造n维数组，n=3;差别过大的像素标注出来且为红色


se=strel('square',20);
im_diff_dilated = imdilate(im_diff,se)
disp(bwconncomp(im_diff_dilated))
im_diff=uint8(im_diff);

im_diff = imlincomb(0.4,im1,10,im_diff,'uint8');
%%计算线性组合0.4*im1+10*im_diff，应该是调节亮度的
figure;
subplot(1,3,1);imshow(im1);
subplot(1,3,2);imshow(im2);
subplot(1,3,3);imshow(im_diff)






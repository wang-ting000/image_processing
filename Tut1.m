%% Activity 2: Spot the Difference 
im = imread('spot_the_difference.png');
im_info = imfinfo('spot_the_difference.png') %�ú������ڻ�ȡһ��ͼƬ�ľ�����Ϣ����Щ������Ϣ����ͼƬ�ĸ�ʽ���ߴ硢��ɫ�������޸�ʱ��ȵ�
im1 = im(:,1:350,:);  
im2 = im(:,351:700,:);  %����ͼ�ֱ�ռ��һ��Ŀ��
im_diff=im1-im2;  %�Ҳ��
im_diff=rgb2gray(im_diff);
im_diff=im_diff>40; %True��Ϊ1��FalseΪ0


im_diff=cat(3,im_diff*255,zeros(size(im_diff)),zeros(size(im_diff))); %����nά���飬n=3;����������ر�ע������Ϊ��ɫ


se=strel('square',20);
imdilate(im_diff,se)
disp(bwconncomp(im_diff))
im_diff=uint8(im_diff);

im_diff = imlincomb(0.4,im1,10,im_diff,'uint8');
%%�����������0.4*im1+10*im_diff��Ӧ���ǵ������ȵ�
figure;
subplot(1,3,1);imshow(im1);
subplot(1,3,2);imshow(im2);
subplot(1,3,3);imshow(im_diff)






%%%%%%%%%%%%%%%%%%%LBP变换后的图像%%%%%%%%%%%%%%%%%%
clear all
clc
I11 = imread('test2.pgm');
SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];
I12=lbp(I11,SP,0,'i');
subplot(1,2,1);
imshow(I11);
title('原图');
subplot(1,2,2);
imshow(I12);
title('LBP');


%        I=imread('rice.png');
%       mapping=getmapping(8,'u2'); 
%        H1=lbp(I,1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood                                  %using uniform patterns
%        subplot(2,1,1),stem(H1);
% 
%        H2=lbp(I);
%        subplot(2,1,2),stem(H2);
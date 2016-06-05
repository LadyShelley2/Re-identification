% I=imread('test.pgm');
% mapping=getmapping(8,'u2'); 
% H1=lbp(I,1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood
%                          %using uniform patterns
% subplot(2,1,1),stem(H1);
% 
% H2=lbp(I);
% subplot(2,1,2),stem(H2);
% 
% SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];
% I2=lbp(I,SP,0,'i');

im = imread('test.pgm');
c  = cont(im,4,16); 
d  = cont(im,4,16,1:500:2000);

figure
subplot(121),imshow(c,[]), title('VAR image')
subplot(122),imshow(d,[]), title('Quantized VAR image')
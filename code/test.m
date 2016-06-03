%% »À¡≥÷ÿœ÷
% faceW = 64; 
% faceH = 64; 
% numPerLine = 11; 
% ShowLine = 15; 
% 
% Y = zeros(faceH*ShowLine,faceW*numPerLine); 
% for i=0:ShowLine-1 
%   	for j=0:numPerLine-1 
%     	Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(fea(i*numPerLine+j+1,:),[faceH,faceW]); 
%   	end 
% end 
% 
% imagesc(Y);colormap(gray);
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU]=pca(fea);
function index = pca_percent(latent,percent)
%latent:特征值列向量
%percent:要划分的百分比，计算表示信息量的百分比
Sum = sum(latent);
index = 1;
tmp_sum = latent(1);
for i=2 : size(latent)
    if(tmp_sum/Sum>percent)
        break;
    else
        tmp_sum = latent(i)+tmp_sum;
        index = index +1;
    end
end
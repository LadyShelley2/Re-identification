function index = pca_percent(latent,percent)
%latent:����ֵ������
%percent:Ҫ���ֵİٷֱȣ������ʾ��Ϣ���İٷֱ�
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
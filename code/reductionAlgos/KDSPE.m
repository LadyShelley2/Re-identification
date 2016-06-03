clear;clc;close all;
load('../datasets/yale/Yale_32x32.mat');
addpath(genpath('../helpers'))
%% pca 降维
full_feature=fea;
[COEFF, SCORE, LATENT] = pca(full_feature);
index = pca_percent(LATENT,0.9);
X = SCORE(:,1:index);
X = X';%将样本信息转换成每一列为一个样本.

%% 一些参数的设定和常数值
class_samples_num = 11;

all_samples_num = size(X,2);
class_num = all_samples_num/class_samples_num;
class_tag = 1;
class_X =X(:,1:class_samples_num);
param_poly.gamma = 1;
param_poly.c = 1;
param_poly.n = 1;
delta = 1.0e+3;
t_collec = [];
h_collec = [];

%% 对每个样本求解最优的t

for i = 1 : all_samples_num
    if(mod(i,class_samples_num)==1 && i ~= 1)%出现样本种类变化时
        class_tag = class_tag + 1;%更新类别编号
        class_X = X(:,(class_tag-1)*class_samples_num+1:class_tag*class_samples_num);%更新该类别的样本 
    end
    
    sample_class_index = mod(i-1,class_samples_num);% 样本在该类中的脚标0-10
    class_X_copy = class_X; %备份的类别样本信息
    class_X(:,sample_class_index+1) = []; %在当前类样本中剔除自身作为训练样本
    
    %求解BijTphixij
    for j = 1: class_samples_num-1
        bijtphixij(j) = kernel_poly(class_X(:,j),X(:,i),param_poly);
    end
    %求解BijTBij
    for k = 1: class_samples_num-1
        for inner_k = 1: class_samples_num-1
            bijtbij(k,inner_k) = kernel_poly(class_X(:,k),class_X(:,inner_k),param_poly);
        end
    end
%     bijtphixij
%     bijtbij

  %在求得的t 元素中加入0元素
  tmp_t = opt_t(bijtphixij',bijtbij);
  if(sample_class_index ==0)
      t = [0;tmp_t(sample_class_index+1:end,:)];
  elseif(sample_class_index == class_samples_num-1)
      t = [tmp_t(1:end,:);0];
  else    
      t = [tmp_t(1:sample_class_index,:);0;tmp_t(sample_class_index+1:end,:)];
  end    
  pre_t = zeros(floor((i-1)/class_samples_num)*class_samples_num,1);
  after_t = zeros((all_samples_num/class_samples_num-ceil((i)/class_samples_num))*class_samples_num,1);
  h = [pre_t;t;after_t];%将t补为h
  class_X = class_X_copy;%恢复备份的类别样本信息
  t_collec = [t_collec t];
  h_collec = [h_collec h];
  
end

%% 对每个样本求解最优的s
s_collec = [];
for i = 1 : all_samples_num
    %clear 变量
    ctphixij = [];
    ctc = [];
    ctci = [];
    class_tag =  floor((i-1)/class_samples_num); %从0开始记
    tmp_X = X;
    tmp_X(:,class_tag * class_samples_num +1 : (class_tag+1)* class_samples_num) =[];
    
    for j = 1:all_samples_num
        ctphixij(j) = kernel_poly(X(:,j),X(:,i),param_poly);
    end
    ctphixij = ctphixij';
    for p = 1:all_samples_num
        for q = 1:all_samples_num
            ctc(p,q) = kernel_poly(X(:,p),X(:,q),param_poly);
        end    
    end
    for p = 1:all_samples_num
        for q = 1 : all_samples_num - class_samples_num
            ctci(p,q) = kernel_poly(X(:,p),tmp_X(:,q),param_poly);
        end
    end
    tmp_s = opt_s(ctphixij,ctc,ctci,h_collec(:,i),delta);
    zero_vacant = zeros(class_samples_num,1);
    if(class_tag ==0)
        s = [zero_vacant;tmp_s];
    elseif(class_tag == class_num-1)
        s = [tmp_s;zero_vacant];
    else   
        s = [tmp_s(1:(class_tag+1)* class_samples_num,:);zero_vacant;tmp_s((class_tag+1)* class_samples_num+1:end,:)];
    end
    s_collec = [s_collec s];
end
%合并s和h 形成最终的权值 d
d_collec = s_collec + t_collec;

%% 求解前d个特征值，将维度降至d维

%% 分类

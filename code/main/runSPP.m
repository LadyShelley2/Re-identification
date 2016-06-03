%% 加载数据
clear;clc;close all;
load('../datasets/yale/Yale_32x32.mat');
addpath(genpath('../helpers'));
addpath(genpath('../reductionAlgos'));
%% pca 分析
sample_num = size(fea,1);
PCAoptions.PCARatio = 0.9;
[eigvector_PCA, eigvalue_PCA, meanData, new_X] = PCA(fea,PCAoptions);
x_0 = zeros(sample_num-1,1);
res = l1eq_pd(x_0, new_X(2:end,:)', new_X(2:end,:)',new_X(1,:)');
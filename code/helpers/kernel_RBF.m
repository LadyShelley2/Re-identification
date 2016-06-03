function [res]= kernel_RBF(v1,v2,params)
%% res = exp(-sqrt((v1-v2)'*(v1-v2))/(sigma*sigma));

%% copied from http://www.kernel-methods.net/matlab/kernels/rbf.m
sig = params.sigma;
coord = v1 - v2;
n=size(coord,1);
K=coord*coord'/sig^2;
d=diag(K);
K=K-ones(n,1)*d'/2;
K=K-d*ones(1,n)/2;
res=exp(K);

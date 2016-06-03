function [res] = kernel_sigmoid(v1,v2,params)
v = params.v;
c = params.c;
res = tanh(v(v1*v2')+c);
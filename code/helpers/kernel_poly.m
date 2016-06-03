function [ res ] = kernel_poly( v1,v2,params )
gamma = params.gamma;
c = params.c;
n = params.n;

res = (gamma*(v1'*v2)+c)^n;


end


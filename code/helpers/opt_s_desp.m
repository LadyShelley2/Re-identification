function x = opt_s_desp(xij,X,Xi,h,delta)

x_0 = zeros(size(Xi,2),1);
Aeq = ones(1,size(Xi,2));
beq = 0;
[x, fval] = fmincon(@(x)func_goal(x),x_0,[],[],Aeq,beq,[],[],@(x)nonlin_con(xij,X,h,Xi, delta,x));

function f = func_goal(s)
f = sum(s.^2);
function [c,ceq]= nonlin_con(xij,X,h,Xi, delta,x)
c = sum((xij-X*h-Xi*x).^2)-delta;
ceq = [];
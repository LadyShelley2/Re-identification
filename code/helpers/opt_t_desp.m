function x = opt_t_desp(xij,Xi)
x_0 = zeros(size(Xi,2),1);
x_0(1) = 1;
A = [];
b = [];
Aeq = ones(1,size(Xi,2));
beq = 1;

[x, fval] = fmincon(@(t)func_goal(xij,Xi,t),x_0,A,b, Aeq, beq);
function f = func_goal(xij,Xi,t)
f = sum((xij-Xi*t).^2);
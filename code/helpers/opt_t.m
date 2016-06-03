function x = opt_t(btphixij,btbi)
x_0 = zeros(size(btbi,2),1);
x_0(1) = 1;
A = [];
b = [];
Aeq = ones(1,size(btbi,2));
beq = 1;
[x, fval] = fmincon(@(t)func_goal(btphixij,btbi,t),x_0,A,b, Aeq, beq);
fval;
function f = func_goal(btphixij,btbi,t )
f = sum((btphixij - btbi * t).^2);
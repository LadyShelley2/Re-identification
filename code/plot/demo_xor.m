xa = -1:0.05:1;
ya = xa;
[x,y]= meshgrid(xa,ya);
z = sign(x.*y);
z(find(z==0))=1;
surf(x,y,z);
figure;
z=[];
z(find(x>=0))=1;
surf(x,y,z);
figure;
z=[];
find()
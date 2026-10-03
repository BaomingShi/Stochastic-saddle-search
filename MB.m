A=[-200, -100, -170, 15];
a = [-1, -1, -6.5, 0.7];
b = [0, 0, 11, 0.6];
c =[-10, -10, -6.5, 0.7];
x= [1, 0, -0.5, -1];
y = [0, 0.5, 1.5, 1];


F=@(uuu) -[sum(A.*exp(a.*(ones(1,4)*uuu(1)-x).^2+b.*(ones(1,4)*uuu(1)-x).*(ones(1,4)*uuu(2)-y)+c.*(ones(1,4)*uuu(2)-y).^2).*(2*a.*(ones(1,4)*uuu(1)-x)+b.*(ones(1,4)*uuu(2)-y)));  sum(A.*exp(a.*(ones(1,4)*uuu(1)-x).^2+b.*(ones(1,4)*uuu(1)-x).*(ones(1,4)*uuu(2)-y)+c.*(ones(1,4)*uuu(2)-y).^2).*(b.*(ones(1,4)*uuu(1)-x)+2*c.*(ones(1,4)*uuu(2)-y)))];   
E=@(uuu) sum(A.*exp(a.*(ones(1,4)*uuu(1)-x).^2+b.*(ones(1,4)*uuu(1)-x).*(ones(1,4)*uuu(2)-y)+c.*(ones(1,4)*uuu(2)-y).^2));

Hessian11=@(uuu) sum(A.*exp(a.*(ones(1,4)*uuu(1)-x).^2+b.*(ones(1,4)*uuu(1)-x).*(ones(1,4)*uuu(2)-y)+c.*(ones(1,4)*uuu(2)-y).^2)*2.*a)+sum(A.*exp(a.*(ones(1,4)*uuu(1)-x).^2+b.*(ones(1,4)*uuu(1)-x).*(ones(1,4)*uuu(2)-y)+c.*(ones(1,4)*uuu(2)-y).^2).*(2*a.*(ones(1,4)*uuu(1)-x)+b.*(ones(1,4)*uuu(2)-y)).^2);   
Hessian12=@(uuu) sum(A.*exp(a.*(ones(1,4)*uuu(1)-x).^2+b.*(ones(1,4)*uuu(1)-x).*(ones(1,4)*uuu(2)-y)+c.*(ones(1,4)*uuu(2)-y).^2)*2.*b)+sum(A.*exp(a.*(ones(1,4)*uuu(1)-x).^2+b.*(ones(1,4)*uuu(1)-x).*(ones(1,4)*uuu(2)-y)+c.*(ones(1,4)*uuu(2)-y).^2).*(2*a.*(ones(1,4)*uuu(1)-x)+b.*(ones(1,4)*uuu(2)-y).*(b.*(ones(1,4)*uuu(1)-x)+2*c.*(ones(1,4)*uuu(2)-y))));   
Hessian22=@(uuu) sum(A.*exp(a.*(ones(1,4)*uuu(1)-x).^2+b.*(ones(1,4)*uuu(1)-x).*(ones(1,4)*uuu(2)-y)+c.*(ones(1,4)*uuu(2)-y).^2)*2.*c)+sum(A.*exp(a.*(ones(1,4)*uuu(1)-x).^2+b.*(ones(1,4)*uuu(1)-x).*(ones(1,4)*uuu(2)-y)+c.*(ones(1,4)*uuu(2)-y).^2).*(b.*(ones(1,4)*uuu(1)-x)+2*c.*(ones(1,4)*uuu(2)-y)).^2);   

Hessian=@(uuu) [Hessian11(uuu),Hessian12(uuu);Hessian12(uuu),Hessian22(uuu)];

u=[-0.4;0.6];



[xdata,ydata]=meshgrid([-1.4:0.01:0.2],[0:0.01:1.8]);

for i=1:length([0:0.01:1.8])
    for j=1:length([-1.4:0.01:0.2])
        zdata(i,j)=E([xdata(i,j);ydata(i,j)]);
    end
end
contour(xdata,ydata,zdata,50);

ustar=[-0.822001557452550;0.624312804305146];
options=struct('k',1,'innpfunc',@(x,y)(x'*y),'maxiter',10000,'dt',0.001);options.outputX=1;options.gammamax=10;options.gammamin=0.5;options.betat=5;options.betau=0.2;
options.V=u;
options.stepsize1=0.01;
options.stepsize2=100;
options.randsize=100;
[u, x_total,output] =  stochastic_hisd_saddle_eigenvector(F,u,options,Hessian);
hold on
plot(x_total(1,1:1:end),x_total(2,1:1:end),'--*','color',[1 0.1 0.3],'LineWidth',1)

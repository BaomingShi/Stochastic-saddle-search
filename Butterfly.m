clc
clear
F=@(uuu) -[4*uuu(1)^3-4*uuu(1)-3*uuu(1)*uuu(2)^2+2*uuu(1)*uuu(2);4*uuu(2)^3+2*uuu(2)-3*uuu(1)^2*uuu(2)+uuu(1)^2-3*uuu(2)^2];

Hessian=@(uuu) [12*uuu(1)^2-4-3*uuu(2)^2+2*uuu(2),-6*uuu(1)*uuu(2)+2*uuu(1);-6*uuu(1)*uuu(2)+2*uuu(1),12*uuu(2)^2+2-3*uuu(1)^2-6*uuu(2)];

 

rng(2)
H = @ (xx,vv,ll) - ( F (xx + ll * vv) - F (xx - ll * vv) ) / (2 * ll);
options=struct('k',1,'innpfunc',@(x,y)(x'*y),'maxiter',10000,'dt',0.001);options.outputX=1;options.gammamax=10;options.gammamin=0.5;options.betat=5;options.betau=0.2;
%%

u=[1.3;-0.7];options.V=u;


[~, x_total,~] = stochastic_hiosd_sirqit_butterfly(F,u,options,Hessian);

[xdata,ydata]=meshgrid([-3:0.01:3],[-2:0.01:3]);

[xquiver,yquiver]=meshgrid([-3:0.3:3],[-2:0.3:3]);
[ynum,xnum]=size(xquiver);
for i=1:xnum
    for j=1:ynum
        xnow=xquiver(1,i);
        ynow=yquiver(j,1);
        hessian=Hessian([xnow;ynow]);
        [eigenvec,~]=eig(hessian);
        eigenvec=eigenvec(:,1);
        gradF=F([xnow;ynow]);
        eigenvec=(eye(2)-2*eigenvec*eigenvec')*gradF;
        eigenvec=eigenvec/norm(eigenvec);
        uu(j,i)=eigenvec(1);
        vv(j,i)=eigenvec(2);
    end
end


contour(xdata,ydata,xdata.^4-2*xdata.^2+ydata.^4+ydata.^2-1.5*xdata.^2.*ydata.^2+xdata.^2.*ydata-ydata.^3,80,'HandleVisibility','off');



hold on
quiver(xquiver,yquiver,uu,vv,0.5,'Color','black','LineWidth',0.8,'MaxHeadSize', 0.2,'HandleVisibility','off')
hold on

plot(x_total(1,1:1:end),x_total(2,1:1:end),'--*','color',[1 0.1 0.3],'LineWidth',1)

hold on
u=[1.3;-0.7];options.V=u;


[x, x_total,output] = hiosd_sirqit_butterfly(F,u,options,Hessian);
plot(x_total(1,1:1:end),x_total(2,1:1:end),'--*','color','blue','LineWidth',1)
hold on
scatter(0, 0, 50, [0.1, 0.6, 0.8], 'o', 'filled', 'LineWidth', 2);
axis([-3,3,-2,3])
set(gca,'fontsize',24)
% legend('Stochastic saddle-search algorithm','Deterministic saddle dynamics','Saddle point')

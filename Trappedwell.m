
F=@(uuu) -[4*uuu(1)*(uuu(1)^2+uuu(2)^2)+2*uuu(1)-1;4*uuu(2)*(uuu(1)^2+uuu(2)^2)-2*uuu(2)+1];   
Hessian=@(uuu) [4*(uuu(1)^2+uuu(2)^2)+8*uuu(1)^2+2,8*uuu(1)*uuu(2);8*uuu(1)*uuu(2),4*(uuu(1)^2+uuu(2)^2)+8*uuu(2)^2-2]; 
H = @ (xx,vv,ll) - ( F (xx + ll * vv) - F (xx - ll * vv) ) / (2 * ll);
options=struct('k',1,'innpfunc',@(x,y)(x'*y),'maxiter',10000,'dt',0.001);options.outputX=1;options.gammamax=10;options.gammamin=0.5;options.betat=5;options.betau=0.2;

u=[1;-1];options.V=u;
options.randsize=1;
options.stepsize1=0.2;

[~, x_total,~] = stochastic_hiosd_sirqit_randf(F,u,options,Hessian);

figure(1)

plot(x_total(1,1:1:end),x_total(2,1:1:end),'--*','color',[0.8 0.2 0.2],'LineWidth',1)

hold on

[xquiver,yquiver]=meshgrid([-1.5:0.1:1.5],[-1.5:0.1:1.5]);
[ynum,xnum]=size(xquiver);
for i=1:xnum
    for j=1:ynum
        xnow=xquiver(1,i);
        ynow=yquiver(j,1);
        hessian=Hessian([xnow;ynow]);
        [eigenvec,~]=eig(hessian);
        eigenvec=eigenvec(:,1);
        gradF=F([xnow;ynow]);
        gradF=(eye(2)-2*eigenvec*eigenvec')*gradF;
        gradF=gradF/norm(gradF);
        uu(j,i)=gradF(1);
        vv(j,i)=gradF(2);
    end
end


quiver(xquiver,yquiver,uu,vv,'Color','black','HandleVisibility','off')
set(gca,'fontsize',24)
axis([-1.5,1.5,-1.5,1.5])
hold on

u=[1;-1];options.V=u;
options.randsize=0;
options.stepsize1=0.01;

[u, x_total,output] = stochastic_hiosd_sirqit_randf(F,u,options,Hessian);


plot(x_total(1,1:1:end),x_total(2,1:1:end),'--*','color',[0.1 0.1 0.8],'LineWidth',1)


function [x, x_total,output] = hiosd_sirqit_butterfly(F,x,options,Hessian)
%% reading parameters
n=length(x);
% innp can compute multi - one innp as a column vector
if isfield(options,'innpfunc')
    innp = @(x,y)options.innpfunc(x,y);
    nor = @(x)sqrt(innp(x,x));
else
    disp('HiOSD: standard inner product.');
    innp = @(x,y)(x'*y);
    nor = @(x)sqrt(x'*x);
end

if isfield(options,'k')
    if isfield(options,'V')
        if options.k ~= size(options.V,2)
            options.k = size(options.V,2);
            disp('HiOSD: k is reset as the column - number of V.');
        end
    end
elseif isfield(options,'V')
    options.k = size(options.V,2);
else
    options.k = 1;
    disp('HiOSD: k is reset as 1.');
end
k = options.k;

V = zeros(n,k);
if isfield(options,'V')
    V(:,1) = options.V(:,1) / nor(options.V(:,1));
    for i = 2 : k
        V(:,i) = options.V(:,i) - V(:,1:i-1) * innp(V(:,1:i-1), options.V(:,i));
        V(:,i) = V(:,i)/nor(V(:,i));
    end
    if rank(V) < k
        [V,~,iter] = hiosd_sirqit_ini(F, x, options);
        disp(['initialize V using sirqit in ' num2str(iter) ' steps'])
    end
else
    [V,~,iter] = hiosd_sirqit_ini(F, x, options);
    disp(['initialize V using sirqit in ' num2str(iter) ' steps'])
end

if isfield(options,'dt')
    dt=options.dt;
else
    dt=0.01;
end

if isfield(options,'l')
    l=options.l;
else
    l=1e-6;
end
if isfield(options,'minl')
    minl=options.minl;
else
    minl=1e-6;
end
if isfield(options,'epsdx')
    epsdx=options.epsdx;
else
    epsdx=1e-12;
end
if isfield(options,'epsf')
    epsf=options.epsf;
else
    epsf=1e-6;
end
if isfield(options,'maxiter')
    maxiter=options.maxiter;
else
    maxiter=1e4;
end
if isfield(options,'betat')
    betat=options.betat;
else
    betat=1;
end
if isfield(options,'betau')
    betau=options.betau;
else
    betau=1;
end
if isfield(options,'tau')
    tau=options.tau;
else
    tau=0.1;
end
if isfield(options,'gammamax')
    gammamax=options.gammamax;
else
    gammamax=1;
end
if isfield(options,'gammamin')
    gammamin=options.gammamin;
else
    gammamin=1;
end
% if ~isfield(options,'outputX')
%     options.outputX=0;
% end
% if options.outputX~=0
%     X=zeros(k+2+k+n,ceil(maxiter/options.outputX));
%     storing='X(:,ceil(iter/options.outputX))=[nor(f);alpha;beta;transpose(gamma);x];';
% else
%     storing='';
% end

%% setting environment
alpha (1 : k, 1) = 0;

f = F (x);
iter=1;
x_total=x;
hessian=Hessian(x);
[eigenvec,~]=eig(hessian);
V=eigenvec(:,1);
E=@(data) data(1).^4-2*data(1).^2+data(2).^4+data(2).^2-1.5*data(1).^2.*data(2).^2+data(1).^2.*data(2)-data(2).^3;
while iter<=maxiter
    f=f;

    % aa=randperm(n,1);
    % f=zeros(n,1);
    % x_tail=x;x_head=x;
    % x_tail(aa)=x(aa)-0.00001;x_head(aa)=x(aa)+0.00001;
    % f(aa)=(E(x_tail)-E(x_head))/(0.00002);

    [eigenvec,~]=eig(Hessian(x));
    V=eigenvec(:,1);
    % V=V+0.1*randn(size(V));
    % V=V/norm(V);
    g = f - 2 * V * innp(V,f);

    Dx = g * 0.5/iter;
    
    x = x + Dx;
    x_total=[x_total x];
    % for i = 1 : k
    %     ui = H (x, V(:,i), l);
    %     alpha(i) = innp (V(:,i), ui);
    %     dV(:,i)= -ui + alpha(i) * V(:,i) + V(:,1:i-1) * ( 2 * innp(V(:,1:i-1),ui) );
    % end
    
    % V = V + dV .* gamma;
    % for i = 1 : k
    %     V(:,i) = V(:,i) - V(:,1:i-1) * innp(V(:,1:i-1), V(:,i));
    %     V(:,i) = V(:,i)/nor(V(:,i));
    % end
    % 
    % l=max(l/(1+dt),minl);

    f = F (x);
    norf = nor(f);
%     eval(storing);
    
    if norf < epsf
%         if options.outputX~=0
%             X=X(:,1:ceil(iter/options.outputX));
%         end
        break;
    end
    iter=iter+1;
end
if iter==maxiter+1
    iter=maxiter;
end
output = struct('x', x, 'V', V, 'it', iter, 'al', alpha);
% if options.outputX~=0
%     output.X=X;
% end

end
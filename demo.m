%% generate data
rng(1);
N = 5;
T = 100;
p = 2;

X = 2*randn(N, p);
BETA = zeros(p,T);
for k = 1:p
    BETA(k,:) = interp1(linspace(0,1,10),randn(10,1)*0.2,linspace(0,1,T),'spline');
end

LAM = exp(X*BETA);
Y = poissrnd(LAM);

figure(1)
subplot(1,2,1)
imagesc(LAM)
colorbar
subplot(1,2,2)
imagesc(Y)
colorbar

%% PG

% hyper-parameter
m0 = zeros(p,1);
V0 = eye(p);
A = eye(p);
b = zeros(p,1);
Sig = eye(p)*1e-2;

% tuning parameter
d = 10;
Rmin = .5;

ng = 1000;
BETA_samp = zeros(p,T,ng);
for k = 1:p
    BETA_samp(k,:,1) =...
        interp1(linspace(0,1,10),randn(10,1)*0.2,linspace(0,1,T),'spline');
end

figure(2)
subplot(1,2,1)
plot(BETA')
title('true')
subplot(1,2,2)
plot(BETA_samp(:,:,1)')
title('sample')

for g = 2:ng
    
    %(1) sample new & original -> new
    [BETA_new, lpdf_oToN] =...
        PG_PLDS_samp(BETA_samp(:,:,g-1),BETA_samp(:,:,g-1),...
        Y,X,A,b,Sig,m0,V0,d,Rmin,true);
    
    %(2) new -> original
    [~, lpdf_nToO] =...
        PG_PLDS_samp(BETA_new,BETA_samp(:,:,g-1),...
        Y,X,A,b,Sig,m0,V0,d,Rmin,false);
    
    % (2) MH step
    lam_ori = exp(X*BETA_samp(:,:,g-1));
    lam_new = exp(X*BETA_new);
    
    lhr = sum(-lam_new + Y.*log(lam_new + (lam_new == 0)), 'all')+...
        prior_beta(BETA_new,A,b,Sig,m0,V0)-...
        sum(-lam_ori + Y.*log(lam_ori + (lam_ori == 0)), 'all')-...
        prior_beta(BETA_samp(:,:,g-1),A,b,Sig,m0,V0)+...
        lpdf_nToO - lpdf_oToN;
    
    if(log(rand) < lhr)
        BETA_samp(:,:,g) = BETA_new;
    else
        BETA_samp(:,:,g) = BETA_samp(:,:,g-1);
    end
    
    figure(2)
    subplot(1,2,1)
    plot(BETA')
    title('true')
    subplot(1,2,2)
    plot(mean(BETA_samp(:,:,round(g/2):g), 3)')
    title("average: "+ round(g/2)+ " to " +g)
    
end



















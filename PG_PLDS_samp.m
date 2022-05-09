function [BETA_b, lpdf] = PG_PLDS_samp(BETA_a, BETA_b,...
    Y,X,A,b,Sig,m0,V0,d,Rmin,active)

% to debug
% BETA_a = BETA_samp(:,:,g-1);
% BETA_b = BETA_samp(:,:,g-1);
% active = true;

p = size(BETA_a,1);
T = size(BETA_a,2);
N = size(Y,1);
lpdf = 0;

%(1) calculate r_{nt}, \hat{w}_{nt}, \Omega_t, \hat{y}_t
%(a) calculate r_{nt}
ETA_tmp = X*BETA_a;
LAM_tmp = exp(ETA_tmp);
C_tmp = (d^2+1)*exp(LAM_tmp);
R_tmp = LAM_tmp.*log(C_tmp).*1./(log(C_tmp) + LAM_tmp.*...
    lambertw((-C_tmp.^(-1./LAM_tmp).*log(C_tmp))./LAM_tmp));
R_tmp(R_tmp< Rmin) = Rmin;

%(b) calculate \hat{w}_{nt}
EW_tmp = ((R_tmp + Y)./(2*(ETA_tmp - log(R_tmp)))).*...
    ((exp(ETA_tmp) - R_tmp)./(exp(ETA_tmp) + R_tmp));

Omega_tmp = zeros(N,N,T);
Yhat_tmp = zeros(N,T);

for t = 1:T
    %(c) calculate \Omega_t
    Omega_tmp(:,:,t) = diag(EW_tmp(:,t));
    
    %(d) calculate yhat_t
    kt_tmp = (Y(:,t) - R_tmp(:,t))/2 + EW_tmp(:,t).*log(R_tmp(:,t));
    Yhat_tmp(:,t) = (Omega_tmp(:,:,t))\kt_tmp;
end

%(2) FF: forward filtering-- calculate m_t, V_t
m_tmp = zeros(p,T);
V_tmp = zeros(p,p,T);

for t = 1:T
    if t == 1
        m_tt_1 = A*m0 + b;
        V_tt_1 = A*V0*A' + Sig;
    else
        m_tt_1 = A*m_tmp(:,t-1) + b;
        V_tt_1 = A*V_tmp(:,:,t-1)*A' + Sig;
    end
    
    Kt = V_tt_1*X'/((X*V_tt_1*X' + inv(Omega_tmp(:,:,t))));
    m_tmp(:,t) = m_tt_1 + Kt*(Yhat_tmp(:,t) - X*m_tt_1);
    V_tmp(:,:,t) = (eye(p) - Kt*X)*V_tt_1;
    
    V_tmp(:,:,t) = (V_tmp(:,:,t) + V_tmp(:,:,t)')/2;
end

%(3) BS: backward sampling
if active
    BETA_b(:,T) = mvnrnd(m_tmp(:,T),V_tmp(:,:,T))';
end
lpdf = lpdf -0.5*(BETA_b(:,T) - m_tmp(:,T))'/...
    (V_tmp(:,:,T))*(BETA_b(:,T) - m_tmp(:,T));


for t = (T-1):-1:1
    Jt = V_tmp(:,:,t)*A'/(A'*V_tmp(:,:,t)*A' + Sig);
    mstar_tmp = m_tmp(:,t) + Jt*(BETA_b(:,t+1) - A*m_tmp(:,t) - b);
    Vstar_tmp = (eye(p) - Jt*A)*V_tmp(:,:,t);
    Vstar_tmp = (Vstar_tmp + Vstar_tmp')/2;
    if active
        BETA_b(:,t) = mvnrnd(mstar_tmp,Vstar_tmp)';
    end
    lpdf = lpdf -0.5*(BETA_b(:,t) - mstar_tmp)'/...
        (Vstar_tmp)*(BETA_b(:,t) - mstar_tmp);
end


end
function lpdf = prior_beta(BETA,A,b,Sig,m0,V0)

% to debug
% BETA = BETA_new;

T = size(BETA,2);
lpdf = -0.5*(BETA(:,1) - m0)'/(V0)*(BETA(:,1) - m0);
for t = 2:T
    lpdf = lpdf -0.5*(BETA(:,t) - A*BETA(:,t-1) - b)'/(Sig)*...
        (BETA(:,t) - A*BETA(:,t-1) - b);
end

end
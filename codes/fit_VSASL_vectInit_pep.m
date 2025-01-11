function [beta, conintval, varargout] = fit_VSASL_vectInit_pep(PLD,diff_sig, Init, T1_artery, T2_factor, alpha_BS1,alpha_VSASL)
% vect: input as a vector from dynamics or different PLD scans
% pep: passing extra parameter, using the nested function method
% preset TI = PLD, TI used in VSASL is same as PLD used in PCASL
% beta: 2x1 [CBF; ATT]
% conintval: 95% confident interval of CBF and ATT

cbf0 = Init(1)/6000; % 50
ATT0 = Init(2); % 1500
beta0=[cbf0 ATT0];
ub=[100/6000 6000]; 
lb=[1/6000 100];   
options=optimset('Display','off'); 
[beta,resnorm,residual,exitflag,output,lambda_fit,jacobian] = lsqcurvefit(@fun_VSASL_1comp_vect_pep, beta0, PLD, diff_sig, lb, ub, options); 
conintval = nlparci(beta,residual,jacobian);
v = length(diff_sig(:)) - length(beta0); % degree of freedom
rmse = norm(residual) / sqrt(v); % variance
varargout{1} = rmse;
varargout{2} = v; 

    function diff_sig = fun_VSASL_1comp_vect_pep(beta, PLD)
        CBF = beta(1);
        ATT = beta(2);
        % t = 1:1:10000; % ms
        diff_sig = zeros(length(PLD),1); % 1: PCASL sig 
        % T1_tissue = 1200;
        % T1_artery = 1850;
        %0.32;
        % T1_app = 1000/(1000/T1_tissue+CBF/lambda);
        % T1_app = (T1_tissue + T1_artery)/2; %1650;       

        M0_b=1;
        lambda = 0.90;

        % VSASL scale factor
        alpha2=alpha_VSASL*(alpha_BS1^3);

        % use simplified equation
        index_1 = logical(PLD <= ATT);
        
        if sum(double(~index_1))>0
            diff_sig(~index_1,1) =  2*M0_b.*CBF*alpha2/lambda.*ATT/1000.*exp(-PLD(~index_1)/T1_artery)*T2_factor; % VSASL: PLD >= ATT
        end
        if sum(double(index_1))>0
            diff_sig(index_1,1) = 2*M0_b.*CBF*alpha2/lambda.*PLD(index_1)/1000.*exp(-PLD(index_1)/T1_artery)*T2_factor; % VSASL: PLD < ATT  
        end
    end
end
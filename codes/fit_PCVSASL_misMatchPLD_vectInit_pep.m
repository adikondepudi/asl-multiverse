function [beta, conintval, varargout] = fit_PCVSASL_misMatchPLD_vectInit_pep(PLDTI,diff_sig, Init, T1_artery, T_tau, T2_factor, alpha_BS1, alpha_PCASL, alpha_VSASL)
% vect: input as a vector from dynamics or from multiple PLDs (different
% PLDs are used between PCASL and VSASL, this only work for multi-delay MULTIVERSE.)
% pep: passing extra parameter, using the nested function method
% preset TI = PLD, TI used in VSASL is same as PLD used in PCASL
% beta: 2x1 [CBF; ATT]
% conintval: 95% confident interval of CBF and ATT

cbf0 = Init(1)/6000; %50
ATT0 = Init(2); %1500
beta0=[cbf0 ATT0];
ub=[100/6000 6000]; 
lb=[1/6000 100];   
options=optimset('Display','off'); 
[beta,resnorm,residual,exitflag,output,lambda_fit,jacobian_M] = lsqcurvefit(@fun_PCVSASL_misMatchPLD_vect_pep, beta0, PLDTI, diff_sig, lb, ub, options); 
conintval = nlparci(beta,residual,jacobian_M);
df = length(diff_sig(:)) - length(beta0); % degree of freedom
rmse = norm(residual) / sqrt(df); % variance
varargout{1} = rmse;
varargout{2} = df;            


    function diff_sig = fun_PCVSASL_misMatchPLD_vect_pep(beta, PLDTI)
        CBF = beta(:,1);
        ATT = beta(:,2);
        PLD = PLDTI(:,1);
        TI = PLDTI(:,2);
        % t = 1:1:10000; % ms
        diff_sig = zeros(length(PLD),2); % 1: PCASL sig 2: VSASL
        % T1_tissue = 1200;
        % T1_artery = 1850;
        %0.32;
        % T1_app = 1000/(1000/T1_tissue+CBF/lambda);
        % T1_app = (T1_tissue + T1_artery)/2; %1650;       

        M0_b=1;
        lambda = 0.90;

%         % VSASL labeling efficiency factor
%         alpha_VSASL = 0.56;

        % VSASL scale factor
        alpha2=alpha_VSASL*(alpha_BS1^3);

%         % PCASL labeling efficiency factor
%         alpha_PCASL = 0.85;

        % PCASL scale factor
        alpha1=alpha_PCASL*(alpha_BS1^4);

        % use simplified equation
        index_1_p = logical(ATT<= PLD);
        index_1_v = logical(ATT <=TI);
        if sum(double(index_1_p))>0
            diff_sig(index_1_p,1) =  2*M0_b.*CBF*alpha1/lambda*T1_artery/1000.*exp(-(PLD(index_1_p))/T1_artery)*(1-exp(-T_tau/T1_artery))*T2_factor; % PCASL: PLD >= ATT            
        end
        if sum(double(index_1_v))>0
            diff_sig(index_1_v,2) = 2*M0_b.*CBF*alpha2/lambda.*ATT/1000.*exp(-TI(index_1_v)/T1_artery)*T2_factor; % VSASL: TI >= ATT
        end
        if sum(double(~index_1_p))>0
            diff_sig(~index_1_p,1) = 2*M0_b.*CBF*alpha1/lambda*T1_artery/1000.*(exp(-ATT./T1_artery)-exp(-(T_tau+PLD(~index_1_p))./T1_artery))*T2_factor; % PCASL: PLD < ATT            
        end   
        if sum(double(~index_1_v))>0
            diff_sig(~index_1_v,2) = 2*M0_b.*CBF*alpha2/lambda.*TI(~index_1_v)/1000.*exp(-TI(~index_1_v)/T1_artery)*T2_factor; % VSASL: TI < ATT
        end
        index_0 = logical(PLD < ATT-T_tau); % PCASL=0: PLD < ATT - Tau
        if sum(double(index_0))>0
            diff_sig(index_0,1) = 0;  % PCASL=0: PLD < ATT - Tau
        end
    end
end

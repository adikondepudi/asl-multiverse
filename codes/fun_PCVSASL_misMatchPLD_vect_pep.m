function diff_sig = fun_PCVSASL_misMatchPLD_vect_pep(beta, PLDTI, T1_artery, T_tau, T2_factor, alpha_BS1, alpha_PCASL, alpha_VSASL)
% vect: input as a vector from dynamics or from multiple PLDs (different

        CBF = beta(:,1);
        ATT = beta(:,2);
        PLD = PLDTI(:,1);
        TI = PLDTI(:,2);

        diff_sig = zeros(length(PLDTI),2); % 1: PCASL sig 2: VSASL
     
        M0_b=1;
        lambda = 0.90;

        % VSASL scale factor
        alpha2=alpha_VSASL*(alpha_BS1^3);

        % PCASL scale factor
        alpha1=alpha_PCASL*(alpha_BS1^4);

        % use simplified equation
        index_1_p = logical(ATT<= PLD);
        index_1_v = logical(ATT <= TI);
        if sum(double(index_1_p))>0
            diff_sig(index_1_p,1) =  2*M0_b.*CBF*alpha1/lambda*T1_artery/1000.*exp(-(PLD(index_1_p))/T1_artery)*(1-exp(-T_tau/T1_artery))*T2_factor; % PCASL: PLD >= ATT            
        end
        if sum(double(index_1_v))>0
            diff_sig(index_1_v,2) = 2*M0_b.*CBF*alpha2/lambda.*ATT/1000.*exp(-TI(index_1_v)/T1_artery)*T2_factor; % VSASL: PLD >= ATT
        end
        if sum(double(~index_1_p))>0
            diff_sig(~index_1_p,1) = 2*M0_b.*CBF*alpha1/lambda*T1_artery/1000.*(exp(-ATT./T1_artery)-exp(-(T_tau+PLD(~index_1_p))./T1_artery))*T2_factor; % PCASL: PLD < ATT            
        end   
        if sum(double(~index_1_v))>0
            diff_sig(~index_1_v,2) = 2*M0_b.*CBF*alpha2/lambda.*TI(~index_1_v)/1000.*exp(-TI(~index_1_v)/T1_artery)*T2_factor; % VSASL: PLD < ATT
        end
        index_0 = logical(PLD < ATT-T_tau); % PCASL=0: PLD < ATT - Tau
        if sum(double(index_0))>0
            diff_sig(index_0,1) = 0;  % PCASL=0: PLD < ATT - Tau
        end
end
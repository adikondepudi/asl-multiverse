function diff_sig = fun_PCASL_1comp_vect_pep(beta, PLD, T1_artery, T_tau, T2_factor, alpha_BS1,alpha_PCASL)
% function of calculate the PCASL genral kinective curve with consideration
% of T2 factor and BS suppression. 
        diff_sig = zeros(size(PLD));
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

        % PCASL scale factor, 4 BS
        alpha1=alpha_PCASL*(alpha_BS1^4);

        % use simplified equation
        index_0 = logical(PLD < ATT-T_tau);
        index_1 = logical((PLD < ATT) & (PLD >= ATT-T_tau));
        index_2 = logical(PLD>=ATT);
        if sum(double(index_0))>0
            diff_sig(index_0,1) = 0;
        end
        if sum(double(index_1))>0
            diff_sig(index_1,1) = 2*M0_b.*CBF*alpha1/lambda*T1_artery/1000.*(exp(-ATT/T1_artery)-exp(-(T_tau+PLD(index_1))./T1_artery))*T2_factor; % PCASL PLD <= ATT   
        end
        if sum(double(index_2))>0
            diff_sig(index_2,1) =  2*M0_b.*CBF*alpha1/lambda*T1_artery/1000.*exp(-(PLD(index_2))/T1_artery)*(1-exp(-T_tau/T1_artery))*T2_factor; % PCASL: PLD > ATT
        end        
end
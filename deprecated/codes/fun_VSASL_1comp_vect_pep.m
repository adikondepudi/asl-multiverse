function diff_sig = fun_VSASL_1comp_vect_pep(beta, PLD,T1_artery, T2_factor, alpha_BS1,alpha_VSASL)
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
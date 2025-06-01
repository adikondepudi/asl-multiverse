% perform grid search for CBF: 1:1:100, ATT: 1:25:4000
% update the noise scalling according to TR based on Qin's suggestion

% This code use TR specific timing to correct the nosie level scaling factor. 
% VSASL PLD=1500ms TR=3936ms
% 	Tsat=2000ms, VS module=64ms, GRASE shot = 240ms, extra time = 3936 - 2000 - 1500 -64 - 240 = 132 ms; 
% 	basetime_VS= 240+64+132 =  436 ms.
% 
% PCASL PLD = 1500ms, Label duration = 1800, TR=4000 ms, extra time = 4000- (1500 + 1800 + 240 ) = 460 ms.
% 	basetime_PC = 240+460 = 700 ms.
% 
% TR for PCASL PLD=2000ms, TR=1800 + 2000 + 240 + 460 = 4500 ms.
% 
% 5-PLD VSASL 500:500:2500, total scan time (1 TR 1 pair 1 repeat) = sum(500:500:2500) + (2000+basetime_VS) * 5 = 19680ms
% 5-PLD PCASL 500:500:2500, total scan time (1 TR 1 pair 1 repeat) = sum(500:500:2500) + (1800+basetime_PC) * 5 = 20000ms
% 
% noise scaling for 5-PLD PCASL = sqrt(20000/4500) = sqrt(4.44)
% noise scaling for 5-PLD VSASL = sqrt(19680/4500) = sqrt(4.37)
% noise scaling for 5-PLD MULTIVERSE = sqrt((19680+20000)/4500) = sqrt(8.82)



% Monte Carlo simulation of combining VSASL and PCASL
% this simulation is to compare PC+VS vs multi-PLD PCASL, single PLD PCASL, VSASL
% for Accuracy defined by Bias, Precison defined by CoV in CBF and ATT
% at different SNR levels, 0.5, 1, 2, 4
% also examine the multi-PLD PCASL and multi-PLD VSASL's fiting property

%ATT ms arterial transit time; aks Bolus Arrival Time (BAT)
%PLD ms Post Labeling Delay in PCASL;
%TI ms inflow time for VSASL; aks bolus duration.


addpath(fullfile(drivedir, '\1Ideas\ASL_model_quan\VSASL kinetic curve\'));
addpath(fullfile(drivedir, '\1Ideas\combineVSASLandPCASL\codes\'));
fullfilename_fitresult = fullfile(drivedir, '\1Ideas\combineVSASLandPCASL\codes\', 'simu_cmpPCVS_Gauss_snr5_cbf20_5PLD_scalNoiseTR_20241230_grids10min.mat');
% General parameters
CBF = 20;
cbf = CBF/6000;
% T1_app = 1600;
T1_artery = 1850;

T_tau = 1800; % PCASL labeling duration
T_sat_vs = 2000; % VSASL Saturation delay
T2_factor=1; % no crusher effect; crusher effect 0.78
alpha_BS1=1; % no background suppression effect; BS effect 0.95
lambda = 0.9;
alpha_PCASL = 0.85;
alpha_VSASL = 0.56;
M0_b=1;
sig_level = fun_PCASL_1comp_vect_pep([CBF/6000;1500], 2000, T1_artery, T_tau, T2_factor, alpha_BS1,alpha_PCASL);

tSNR = 5;
noiseSD_single  = sig_level/tSNR;% signal level =0.0082 when T_tau=1800, i_PLD=i_TI=1500, T2_factor=1, alpha_BS1=1; 
N_noise = 1000;%times of noise generation
noisemean = 0;

ATT = 0:100:4000;
%--------------------------------------------------
% generate VSASL 1PLD, PCASL 1PLD, PCASL mPLD, and VSASL mPLD signal
% PLD_combine = {[1000 1000], [1500 1500], [1000 1000;2000 2000], [1000 1000; 1500 1500; 2000 2000;], [1000 1000; 1500 1500; 2000 2000; 2500 2500], [500 500; 1000 1000; 1500 1500; 2000 2000;2500 2500],...
    % [500 500], [2000 2000],[2500 2500]};
PLD_combine = {[500 500; 1000 1000; 1500 1500; 2000 2000;2500 2500]};
scantime_PCASL_PLD2000 = 0.6;
scantime_PCASL_mPLD = 2.86;
scantime_VSASL_mPLD = 2.63;
scantime_PCVS = 2.86+2.63;



% fitting result
CBF_pcvs_est = zeros(N_noise, length(ATT), length(PLD_combine)); % [diff PLD combination;]
ATT_pcvs_est = zeros(N_noise, length(ATT), length(PLD_combine));% [diff PLD combination;]
conintval_pcvs_est = zeros(N_noise, length(ATT), 2, length(PLD_combine));
rmse_pcvs_est = zeros(N_noise, length(ATT), length(PLD_combine));
CBF_PCASL_mPLD_est = zeros(N_noise, length(ATT),length(PLD_combine)); 
ATT_PCASL_mPLD_est = zeros(N_noise, length(ATT), length(PLD_combine)); 
conintval_PCASL_mPLD = zeros(N_noise, length(ATT), 2, length(PLD_combine));
rmse_PCASL_mPLD = zeros(N_noise, length(ATT), length(PLD_combine));
CBF_VSASL_mPLD_est = zeros(N_noise, length(ATT), length(PLD_combine)); 
ATT_VSASL_mPLD_est = zeros(N_noise, length(ATT), length(PLD_combine)); 
conintval_VSASL_mPLD = zeros(N_noise, length(ATT), 2, length(PLD_combine));
rmse_VSASL_mPLD  = zeros(N_noise, length(ATT), length(PLD_combine));
CBF_PCASL_1pld_est = zeros(N_noise, length(ATT),length(PLD_combine)); 
CBF_VSASL_1pld_est = zeros(N_noise, length(ATT),length(PLD_combine)); 

record_Init_allthree = zeros(N_noise, length(ATT), 2, 3, length(PLD_combine));

x0 = [50/6000; 1500];
ub=[100/6000; 6000]; 
lb=[1/6000; 100]; 
options=optimset('Display','off'); 

for pp = 1:length(PLD_combine)

    % PLD = 500:500:2500;
    PLD1 = PLD_combine{pp}(:,1);
    PLD = PLD_combine{pp}(~isnan(PLD1),1);
    TI1 = PLD_combine{pp}(:,2);
    TI = PLD_combine{pp}(~isnan(TI1), 2);


    noiseSD_mPLD_PCASL_scale = sqrt(scantime_PCASL_mPLD/scantime_PCASL_PLD2000);
    noiseSD_mPLD_VSASL_scale = sqrt(scantime_VSASL_mPLD/scantime_PCASL_PLD2000);
    noiseSD_mPLD_pcvs_scale = sqrt(scantime_PCVS/scantime_PCASL_PLD2000);


    % simulated samples
    sig_VSASL_TI = zeros(length(ATT), length(TI));
    sig_PCASL_PLD = zeros(length(ATT), length(PLD));
    signoise_VSASL_1TI = zeros(N_noise, length(ATT), length(TI));
    signoise_PCASL_1PLD = zeros(N_noise, length(ATT), length(PLD));
    signoise_PCASL_pcvs = zeros(N_noise, length(ATT), length(PLD));
    signoise_VSASL_pcvs = zeros(N_noise, length(ATT), length(TI));
    signoise_PCASL_multiPLD = zeros(N_noise, length(ATT), length(PLD));
    signoise_VSASL_multiTI = zeros(N_noise, length(ATT), length(TI));

    for jj = 1:length(ATT)
        i_ATT = ATT(jj);
        %         tc_VSASL = VSASL_GKM_1comp(cbf, i_ATT, T1_artery, T2_factor, alpha_BS1);
        %         sig_VSASL_TI(jj, ii) = tc_VSASL(i_TI);
        sig_VSASL_TI(jj, :) = fun_VSASL_1comp_vect_pep([cbf;i_ATT], TI, T1_artery, T2_factor, alpha_BS1, alpha_VSASL);
        noise_VSASL = noiseSD_single.*randn([N_noise length(TI)]);
        signoise_VSASL_1TI(:,jj,:) = reshape(repmat(sig_VSASL_TI(jj, :), [N_noise 1]),[N_noise 1 length(TI)] )+ reshape(noise_VSASL, [N_noise 1 length(TI)]);

        %         tc_PCASL = PCASL_GKM_1comp(cbf, i_ATT, T_tau,T1_artery, T2_factor, alpha_BS1);
        %         sig_PCASL_PLD(jj,ii) = tc_PCASL(i_PLD+T_tau);
        sig_PCASL_PLD(jj,:) = fun_PCASL_1comp_vect_pep([cbf; i_ATT], PLD, T1_artery, T_tau, T2_factor, alpha_BS1, alpha_PCASL);
        noise_PCASL = noiseSD_single.*randn([N_noise length(PLD)]);
        signoise_PCASL_1PLD(:,jj,:) = reshape(repmat(sig_PCASL_PLD(jj, :), [N_noise 1]), [N_noise 1 length(PLD)] ) + reshape(noise_PCASL,[N_noise 1 length(PLD)]);

        signoise_PCASL_pcvs(:,jj,:) = reshape(repmat(sig_PCASL_PLD(jj, :), [N_noise 1]), [N_noise 1 length(PLD)]) + reshape(noiseSD_mPLD_pcvs_scale*noise_PCASL, [N_noise 1 length(PLD)]);
        signoise_VSASL_pcvs(:,jj,:) = reshape(repmat(sig_VSASL_TI(jj, :), [N_noise 1]), [N_noise 1 length(TI)]) + reshape(noiseSD_mPLD_pcvs_scale*noise_VSASL, [N_noise 1 length(TI)]);

        signoise_PCASL_multiPLD(:,jj,:) = reshape(repmat(sig_PCASL_PLD(jj, :), [N_noise 1]), [N_noise 1 length(PLD)]) + reshape(noiseSD_mPLD_PCASL_scale*noise_PCASL, [N_noise 1 length(PLD)]);
        signoise_VSASL_multiTI(:,jj,:) = reshape(repmat(sig_VSASL_TI(jj, :), [N_noise 1]),[N_noise 1 length(TI)]) + reshape(noiseSD_mPLD_VSASL_scale*noise_VSASL, [N_noise 1 length(TI)]);
    end
    for jj = 1:length(ATT)
        i_ATT = ATT(jj);
        %         tc_VSASL = VSASL_GKM_1comp(cbf, i_ATT, T1_artery, T2_factor, alpha_BS1);
        %         sig_VSASL_TI(jj, ii) = tc_VSASL(i_TI);
        sig_VSASL_TI(jj, :) = fun_VSASL_1comp_vect_pep([cbf;i_ATT], TI, T1_artery, T2_factor, alpha_BS1, alpha_VSASL);
        noise_VSASL = noiseSD_single.*randn([N_noise length(TI)]);
        signoise_VSASL_1TI(:,jj,:) = reshape(repmat(sig_VSASL_TI(jj, :), [N_noise 1]),[N_noise 1 length(TI)] )+ reshape(noise_VSASL, [N_noise 1 length(TI)]);

        %         tc_PCASL = PCASL_GKM_1comp(cbf, i_ATT, T_tau,T1_artery, T2_factor, alpha_BS1);
        %         sig_PCASL_PLD(jj,ii) = tc_PCASL(i_PLD+T_tau);
        sig_PCASL_PLD(jj,:) = fun_PCASL_1comp_vect_pep([cbf; i_ATT], PLD, T1_artery, T_tau, T2_factor, alpha_BS1, alpha_PCASL);
        noise_PCASL = noiseSD_single.*randn([N_noise length(PLD)]);
        signoise_PCASL_1PLD(:,jj,:) = reshape(repmat(sig_PCASL_PLD(jj, :), [N_noise 1]), [N_noise 1 length(PLD)] ) + reshape(noise_PCASL,[N_noise 1 length(PLD)]);

        signoise_PCASL_pcvs(:,jj,:) = reshape(repmat(sig_PCASL_PLD(jj, :), [N_noise 1]), [N_noise 1 length(PLD)]) + reshape(noiseSD_mPLD_pcvs_scale*noise_PCASL, [N_noise 1 length(PLD)]);
        signoise_VSASL_pcvs(:,jj,:) = reshape(repmat(sig_VSASL_TI(jj, :), [N_noise 1]), [N_noise 1 length(TI)]) + reshape(noiseSD_mPLD_pcvs_scale*noise_VSASL, [N_noise 1 length(TI)]);

        signoise_PCASL_multiPLD(:,jj,:) = reshape(repmat(sig_PCASL_PLD(jj, :), [N_noise 1]), [N_noise 1 length(PLD)]) + reshape(noiseSD_mPLD_PCASL_scale*noise_PCASL, [N_noise 1 length(PLD)]);
        signoise_VSASL_multiTI(:,jj,:) = reshape(repmat(sig_VSASL_TI(jj, :), [N_noise 1]),[N_noise 1 length(TI)]) + reshape(noiseSD_mPLD_VSASL_scale*noise_VSASL, [N_noise 1 length(TI)]);
    end

    % prep grid search 
    grid_CBF = 10:10:90;
    grid_ATT = 100:100:5400;
    % grid_CBF = 30;
    % grid_ATT = 1500;
    grid_sig_VSASL_TI = zeros(length(grid_CBF), length(grid_ATT), length(TI));
    grid_sig_PCASL_PLD = zeros(length(grid_CBF), length(grid_ATT), length(PLD));

    for ii = 1:length(grid_CBF)
        i_CBF = grid_CBF(ii);
        for jj = 1:length(grid_ATT)
            i_ATT = grid_ATT(jj);
            grid_sig_PCASL_PLD(ii,jj,:) =  fun_PCASL_1comp_vect_pep([i_CBF/6000; i_ATT], PLD, T1_artery, T_tau, T2_factor, alpha_BS1, alpha_PCASL);
            grid_sig_VSASL_TI(ii,jj,:) = fun_VSASL_1comp_vect_pep([i_CBF/6000;i_ATT], TI, T1_artery, T2_factor, alpha_BS1, alpha_VSASL);
        end
    end

    for jj = 1:length(ATT)
        for kk = 1:N_noise
            diff_sig_1 = NaN.* ones(size(PLD_combine{pp}));
            diff_sig_1(~isnan(PLD1),1) = squeeze(signoise_PCASL_pcvs(kk,jj,:));
            diff_sig_1(~isnan(TI1),2) = squeeze(signoise_VSASL_pcvs(kk,jj,:));
            PLDTI =  PLD_combine{pp};

            % if use PC+VS ASL model
            % can not do arbitery number of PLD and TI, just do the same
            % number, with match or mismatch PLD-TI
            tmp = reshape(diff_sig_1, [1 1 size(diff_sig_1,1) size(diff_sig_1,2)]);
            grid_L2_MV = sqrt(sum(sum((cat(4,grid_sig_PCASL_PLD, grid_sig_VSASL_TI) - repmat(tmp, [length(grid_CBF) length(grid_ATT) 1 1])).^2, 3),4)/length(diff_sig_1(:)));
            [I,J]=find(grid_L2_MV==min(grid_L2_MV(:)),1,'first');
            Init(1) = grid_CBF(I);
            Init(2) = grid_ATT(J);
            [beta, conintval,rmse] = fit_PCVSASL_misMatchPLD_vectInit_pep(PLDTI, diff_sig_1, Init, T1_artery, T_tau, T2_factor, alpha_BS1, alpha_PCASL, alpha_VSASL);
            fit_cbf = beta(1)*6000;
            fit_bolus = beta(2);
            CBF_pcvs_est(kk,jj, pp) = fit_cbf;
            ATT_pcvs_est(kk,jj, pp) = fit_bolus;
            conintval_pcvs_est(kk,jj,1,pp) = (conintval(1,2)-conintval(1,1))*6000;
            conintval_pcvs_est(kk,jj,2,pp) = conintval(2,2)-conintval(2,1);
            rmse_pcvs_est(kk,jj,pp) = rmse;
            record_Init_allthree(kk, jj, :, 1, pp) = Init(:);

            % for multi-PLD PCASL
            diff_sig_1 = squeeze(signoise_PCASL_multiPLD(kk,jj,:));
            tmp = reshape(diff_sig_1, [1 1 size(diff_sig_1,1)]);
            grid_L2_PCASL = sqrt(sum((grid_sig_PCASL_PLD - repmat(tmp, [length(grid_CBF) length(grid_ATT) 1])).^2, 3)/length(diff_sig_1(:)));
            [I,J]=find(grid_L2_PCASL==min(grid_L2_PCASL(:)),1,'first');
            Init(1) = grid_CBF(I);
            Init(2) = grid_ATT(J);

            [beta, conintval, rmse] = fit_PCASL_vectInit_pep(PLD(:), diff_sig_1,Init, T1_artery, T_tau, T2_factor, alpha_BS1,alpha_PCASL);
            fit_cbf = beta(1)*6000;
            fit_bolus = beta(2);
            CBF_PCASL_mPLD_est(kk, jj,pp) = fit_cbf;
            ATT_PCASL_mPLD_est(kk, jj,pp) = fit_bolus;
            conintval_PCASL_mPLD(kk,jj,1,pp) = (conintval(1,2)-conintval(1,1))*6000;
            conintval_PCASL_mPLD(kk,jj,2,pp) = conintval(2,2)-conintval(2,1);
            rmse_PCASL_mPLD(kk,jj,pp) = rmse;
            record_Init_allthree(kk, jj, :, 2, pp) = Init(:);
            % % compute R2 of the fit
            % yhat = fun_PCASL_1comp_vect_pep(beta, myPLD, T1_artery, T_tau, T2_factor, alpha_BS1,alpha_PCASL);
            % R2fit_PCASL_mPLD(kk, jj) = 1-sum((diff_sig_1-yhat).^2)/(var(diff_sig_1).*(length(diff_sig_1)-1));
            % R2adj_PCASL_mPLD(kk, jj) = 1-sum((diff_sig_1-yhat).^2)/(length(yhat)-length(beta)+1e-5)/var(diff_sig_1);

            % for multi-PLD VSASL
            diff_sig_1 = squeeze(signoise_VSASL_multiTI(kk,jj,:));
            tmp = reshape(diff_sig_1, [1 1 size(diff_sig_1,1)]);
            grid_L2_VSASL = sqrt(sum((grid_sig_VSASL_TI - repmat(tmp, [length(grid_CBF) length(grid_ATT) 1])).^2, 3)/length(diff_sig_1(:)));
            [I,J]=find(grid_L2_VSASL==min(grid_L2_VSASL(:)),1,'first');
            Init(1) = grid_CBF(I);
            Init(2) = grid_ATT(J);

            [beta, conintval, rmse] = fit_VSASL_vectInit_pep(TI(:), diff_sig_1, Init, T1_artery, T2_factor, alpha_BS1,alpha_VSASL);
            fit_cbf = beta(1)*6000;
            fit_bolus = beta(2);
            CBF_VSASL_mPLD_est(kk, jj,pp) = fit_cbf;
            ATT_VSASL_mPLD_est(kk, jj,pp) = fit_bolus;
            conintval_VSASL_mPLD(kk,jj,1,pp) = (conintval(1,2)-conintval(1,1))*6000;
            conintval_VSASL_mPLD(kk,jj,2,pp) = conintval(2,2)-conintval(2,1);
            rmse_VSASL_mPLD(kk,jj,pp) = rmse;
            record_Init_allthree(kk, jj, :, 3, pp) = Init(:);
            % % compute R2 of the fit
            % yhat = fun_VSASL_1comp_vect_pep(beta, myPLD, T1_artery, T2_factor, alpha_BS1,alpha_VSASL);
            % R2fit_VSASL_mPLD(kk, jj) = 1-sum((diff_sig_1-yhat).^2)/(var(diff_sig_1).*(length(diff_sig_1)-1));
            % R2adj_VSASL_mPLD(kk, jj) = 1-sum((diff_sig_1-yhat).^2)/(length(yhat)-length(beta)+1e-5)/var(diff_sig_1);

        end
    end

    for jj = 1:length(ATT)
        for kk = 1:N_noise
            for ii = 1:length(PLD)
                i_PLD = PLD(ii);
                % if use PCASL model
                CBF_PCASL_1pld_est(kk, jj, ii) = 6000*0.5*signoise_PCASL_1PLD(kk,jj,ii)/alpha_PCASL*lambda/T1_artery*1000/exp(-i_PLD/T1_artery)/(1-exp(-T_tau/T1_artery))/T2_factor;
                i_TI = i_PLD;
                % if use VSASL model
                CBF_VSASL_1pld_est(kk, jj, ii) = 6000*0.5*signoise_VSASL_1TI(kk,jj,ii)/alpha_VSASL*lambda/i_TI*1000/exp(-i_TI/T1_artery)/T2_factor;
            end
        end
    end
end
if exist(fullfilename_fitresult, "file") >0     
    save(fullfilename_fitresult, 'ATT_VSASL_mPLD_est','CBF_VSASL_mPLD_est',...
    'ATT_PCASL_mPLD_est','CBF_PCASL_mPLD_est', 'CBF_pcvs_est', 'ATT_pcvs_est','CBF_PCASL_1pld_est','CBF_VSASL_1pld_est',...
     'conintval_pcvs_est','conintval_PCASL_mPLD', 'conintval_VSASL_mPLD','rmse_VSASL_mPLD', 'rmse_PCASL_mPLD', 'rmse_pcvs_est',...
     'record_Init_allthree', '-append');
    
else
    save(fullfilename_fitresult, 'ATT_VSASL_mPLD_est','CBF_VSASL_mPLD_est',...
    'ATT_PCASL_mPLD_est','CBF_PCASL_mPLD_est', 'CBF_pcvs_est', 'ATT_pcvs_est','CBF_PCASL_1pld_est','CBF_VSASL_1pld_est',...
    'conintval_pcvs_est','conintval_PCASL_mPLD', 'conintval_VSASL_mPLD','rmse_VSASL_mPLD', 'rmse_PCASL_mPLD', 'rmse_pcvs_est',...
    'record_Init_allthree');
end

save(fullfilename_fitresult, 'PLD_combine', 'N_noise','tSNR','CBF','ATT','-append');

%------------------simulation is done---------------------------
% 
%--------------- plot for CI and R2------------------------------
load(fullfilename_fitresult);
figure; hold on;
plot(ATT, squeeze(mean(rmse_pcvs_est(:,:,1))), 'r');
plot(ATT, squeeze(mean(rmse_PCASL_mPLD)), 'b');
plot(ATT, squeeze(mean(rmse_VSASL_mPLD)), 'g');
legend('MULTIVERSE 5-PLD', 'PCASL 5-PLD', 'VSASL 5-PLD');
xlabel('ATT (ms)'); ylabel('RMSE = norm(resid)/df');
%------------------------------------------

% calculate bias, CoV, RMSE
load(fullfilename_fitresult);

N_noise = 1000;%times of noise generation
% ATT = 0:200:4000;
% for pp = 1:length(PLD_combine)
% PLD = PLD_combine{pp}(:,1);
% CBF = 60;
% PLD_combine = {[500 500; 1000 1000; 1500 1500; 2000 2000;2500 2500]};
% PLD_index = {[1 1; 2 2; 3 3; 4 4; 5 5]};

ATT_vec = repmat(ATT, N_noise,1);
CBFBIAS  = squeeze(mean(CBF_pcvs_est-CBF*ones(size(CBF_pcvs_est))));
ATTBIAS  = squeeze(mean(ATT_pcvs_est-repmat(ATT_vec, 1, 1, length(PLD_combine))));
CBFBIAS_PCASL_mPLD = squeeze(mean(CBF_PCASL_mPLD_est-CBF*ones(size(CBF_PCASL_mPLD_est))));
ATTBIAS_PCASL_mPLD = squeeze(mean(ATT_PCASL_mPLD_est-repmat(ATT_vec, 1)));
CBFBIAS_VSASL_mPLD = squeeze(mean(CBF_VSASL_mPLD_est-CBF*ones(size(CBF_VSASL_mPLD_est))));
ATTBIAS_VSASL_mPLD = squeeze(mean(ATT_VSASL_mPLD_est-repmat(ATT_vec, 1)));
CBFBIAS_PCASL_1pld = squeeze(mean(CBF_PCASL_1pld_est-CBF*ones(size(CBF_PCASL_1pld_est))));
CBFBIAS_VSASL_1pld = squeeze(mean(CBF_VSASL_1pld_est-CBF*ones(size(CBF_VSASL_1pld_est))));


CBFCoV = squeeze(std(CBF_pcvs_est,0,1, 'omitnan')./mean(CBF_pcvs_est, 1, 'omitnan'));
ATTCoV = squeeze(std(ATT_pcvs_est,0,1, 'omitnan')./mean(ATT_pcvs_est, 1, 'omitnan'));
CBFCoV_PCASL_mPLD = squeeze(std(CBF_PCASL_mPLD_est,0,1, 'omitnan')./mean(CBF_PCASL_mPLD_est, 1, 'omitnan'));
CBFCoV_VSASL_mPLD = squeeze(std(CBF_VSASL_mPLD_est,0,1, 'omitnan')./mean(CBF_VSASL_mPLD_est, 1, 'omitnan'));
CBFCoV_PCASL_1pld = squeeze(std(CBF_PCASL_1pld_est,0,1, 'omitnan')./abs(mean(CBF_PCASL_1pld_est, 1, 'omitnan')));
CBFCoV_VSASL_1pld = squeeze(std(CBF_VSASL_1pld_est,0,1, 'omitnan')./abs(mean(CBF_VSASL_1pld_est, 1, 'omitnan')));
ATTCoV_PCASL_mPLD = squeeze(std(ATT_PCASL_mPLD_est,0,1, 'omitnan')./mean(ATT_PCASL_mPLD_est, 1, 'omitnan'));
ATTCoV_VSASL_mPLD = squeeze(std(ATT_VSASL_mPLD_est,0,1, 'omitnan')./mean(ATT_VSASL_mPLD_est, 1, 'omitnan'));

CBFRMSE = squeeze(sqrt(sum((CBF_pcvs_est - CBF*ones(size(CBF_pcvs_est))).^2./N_noise, 1,'omitnan')));
ATTRMSE = squeeze(sqrt(sum((ATT_pcvs_est - repmat(ATT_vec, 1, 1, length(PLD_combine))).^2./N_noise, 1, 'omitnan')));
CBFRMSE_PCASL_mPLD = squeeze(sqrt(sum((CBF_PCASL_mPLD_est-CBF*ones(size(CBF_PCASL_mPLD_est))).^2./N_noise, 1,'omitnan')));
CBFRMSE_VSASL_mPLD = squeeze(sqrt(sum((CBF_VSASL_mPLD_est-CBF*ones(size(CBF_VSASL_mPLD_est))).^2./N_noise , 1,'omitnan')));
ATTRMSE_PCASL_mPLD = squeeze(sqrt(sum((ATT_PCASL_mPLD_est-repmat(ATT_vec, 1, 1)).^2./N_noise , 1,'omitnan')));
ATTRMSE_VSASL_mPLD = squeeze(sqrt(sum((ATT_VSASL_mPLD_est-repmat(ATT_vec, 1, 1)).^2./N_noise , 1,'omitnan')));
CBFRMSE_PCASL_1pld = squeeze(sqrt(sum((CBF_PCASL_1pld_est-CBF*ones(size(CBF_PCASL_1pld_est))).^2./N_noise,1,'omitnan')));
CBFRMSE_VSASL_1pld = squeeze(sqrt(sum((CBF_VSASL_1pld_est-CBF*ones(size(CBF_VSASL_1pld_est))).^2./N_noise,1, 'omitnan')));

ATTSTD = squeeze(std(ATT_pcvs_est,0,1, 'omitnan'));
ATTSTD_PCASL_mPLD = squeeze(std(ATT_PCASL_mPLD_est,0,1, 'omitnan'));
ATTSTD_VSASL_mPLD = squeeze(std(ATT_VSASL_mPLD_est,0,1, 'omitnan'));

CBFSTD = squeeze(std(CBF_pcvs_est,0,1, 'omitnan'));
CBFSTD_PCASL_mPLD = squeeze(std(CBF_PCASL_mPLD_est,0,1, 'omitnan'));
CBFSTD_VSASL_mPLD = squeeze(std(CBF_VSASL_mPLD_est,0,1, 'omitnan'));




%--------------plot for bias, CoV, RMSE----------------------

% percentage display for bias and CoV
%-----------------accuracy: BIAS----------------------
%Figure 1: BIAS (a) CBF (b) ATT 
figure('Units','normalized','Position', [0, 0, 0.8, 1]);
subplot(3,2,1); hold on; grid on;
plot(ATT, CBFBIAS_PCASL_1pld(:,4)'/CBF*100,'b--','LineWidth', 3);
plot(ATT, CBFBIAS_VSASL_1pld(:,3)'/CBF*100,'g--','LineWidth',3);
plot(ATT, CBFBIAS_PCASL_mPLD/CBF*100,'b-','LineWidth',5); 
plot(ATT, CBFBIAS_VSASL_mPLD/CBF*100,'g-','LineWidth',5); 
% plot(ATT, CBFBIAS(:,2)'/CBF*100,'m-','LineWidth',5); 
plot(ATT, CBFBIAS'/CBF*100,'r-','LineWidth',3); 
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
ylim([-120 20]); yticks([-120:40:20]);ytickformat(gca, 'percentage');
xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)'); 
ylabel('nBias (%)'); 
set(gca,'FontSize',20, 'fontweight','bold'); box on;
% title('fitted CBF','FontSize',20,'FontWeight','bold');

subplot(3,2,2); hold on; grid on;
plot(ATT, ATTBIAS_PCASL_mPLD./(ATT+1)*100,'b-','LineWidth',5);
plot(ATT, ATTBIAS_VSASL_mPLD./(ATT+1)*100,'g-','LineWidth',5);
% plot(ATT, ATTBIAS(:,2)'./(ATT+1)*100,'m-','LineWidth',5);
plot(ATT, ATTBIAS./(ATT+1)*100,'r-','LineWidth',3);
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
xlim([0,4000]); xticks(0:1000:4000);
ylim([-50 200]);yticks(-50:50:200);ytickformat(gca, 'percentage');
% xlabel('true ATT (ms)'); 
ylabel('nBias (%)'); 
set(gca,'FontSize',20,'fontweight','bold'); box on
% title('fitted ATT','FontSize',20,'FontWeight','bold');

%-------------------Precision: CoV-----------------------------
% Figure 2: CoV of (A) CBF  (B) ATT
subplot(3,2,3); hold on; grid on;
plot(ATT, CBFCoV_PCASL_1pld(:,4)*100, 'b--','LineWidth',3);
plot(ATT, CBFCoV_VSASL_1pld(:,3)*100, 'g--','LineWidth',3);
plot(ATT, CBFCoV_PCASL_mPLD*100,'b-','LineWidth',5); 
plot(ATT, CBFCoV_VSASL_mPLD*100,'g-','LineWidth',5);
% plot(ATT, CBFCoV(:,2)'*100,'m-','LineWidth',5); 
plot(ATT, CBFCoV'*100,'r-','LineWidth',3); 
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
ylim([0 150]); yticks(0:50:150);ytickformat(gca, 'percentage');
xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)'); 
ylabel('CoV (%)'); 
set(gca,'FontSize',20, 'FontWeight','bold'); box on
% title('fitted CBF','FontSize',20,'FontWeight','bold');

subplot(3,2,4); hold on; grid on;
plot(ATT, ATTCoV_PCASL_mPLD*100,'b-','LineWidth',5);
plot(ATT, ATTCoV_VSASL_mPLD*100,'g-','LineWidth',5);
% plot(ATT, ATTCoV(:,2)'*100,'m-','LineWidth',5);
plot(ATT, ATTCoV*100,'r-','LineWidth',3);
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
xlim([0, 4000]);xticks(0:1000:4000);
ylim([0 120]); yticks(0:40:120);ytickformat(gca, 'percentage');
% xlabel('true ATT (ms)'); 
ylabel('CoV (%)'); 
set(gca,'FontSize',20,'FontWeight','bold'); box on
% title('fitted ATT','FontSize',20,'FontWeight','bold');

%-------------------Acuracy + Precision: RMSE-----------------------------
%  Figure 3: RSME of (A) CBF  (B) ATT
subplot(3,2,5); hold on; grid on;
plot(ATT, CBFRMSE_PCASL_1pld(:,4)/CBF*100, 'b--','LineWidth',3);
plot(ATT, CBFRMSE_VSASL_1pld(:,3)/CBF*100, 'g--','LineWidth',3);
plot(ATT, CBFRMSE_PCASL_mPLD/CBF*100, 'b-','LineWidth',5);
plot(ATT, CBFRMSE_VSASL_mPLD/CBF*100, 'g-','LineWidth',5);
% plot(ATT, CBFRMSE(:,2)'/CBF*100, 'm-','LineWidth',5);
plot(ATT, CBFRMSE/CBF*100, 'r-','LineWidth',3);
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
xlim([0,4000]);xticks(0:1000:4000);
ylim([0 150]); yticks(0:50:150);ytickformat(gca, 'percentage');
xlabel('true ATT (ms)'); 
ylabel('nRMSE (%)'); 
set(gca,'FontSize',20,'FontWeight','bold'); box on
% title('fitted CBF','FontSize',20,'FontWeight','bold');
% legend({'PCASL PLD=2000ms', 'VSASL PLD=1500ms', 'PCASL multi-PLD','VSASL multi-PLD','MULTIVERSE'},'FontSize',16,'Location','north');


subplot(3,2,6); hold on; grid on;
plot(ATT, ATTRMSE_PCASL_mPLD./(ATT+1)*100, 'b-','LineWidth',5);
plot(ATT, ATTRMSE_VSASL_mPLD./(ATT+1)*100, 'g-','LineWidth',5);
% plot(ATT, ATTRMSE(:,2)'./(ATT+1)*100, 'm-','LineWidth',5);
plot(ATT, ATTRMSE./(ATT+1)*100, 'r-','LineWidth',3);
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
xlim([0,4000]);xticks(0:1000:4000);
ylim([0 300]); yticks(0:100:300);ytickformat(gca, 'percentage');
xlabel('true ATT (ms)'); 
ylabel('nRMSE (%)'); 
set(gca,'FontSize',20,'FontWeight','bold'); box on
% title('fitted ATT','FontSize',20,'FontWeight','bold');
hold off;
%-----------------end plot--------------------------------


%-----------------------------------------------------------
% absolute Bias, SD and RMSE on Y axis
%-----------------------------------------------------------
CBFSD = squeeze(std(CBF_pcvs_est-CBF,0,1, 'omitnan'));
ATTSD = squeeze(std(ATT_pcvs_est-repmat(ATT_vec,[1 1 1]),0,1, 'omitnan'));
CBFSD_PCASL_mPLD = squeeze(std(CBF_PCASL_mPLD_est-CBF,0,1, 'omitnan'));
CBFSD_VSASL_mPLD = squeeze(std(CBF_VSASL_mPLD_est-CBF,0,1, 'omitnan'));
CBFSD_PCASL_1pld = squeeze(std(CBF_PCASL_1pld_est-CBF,0,1, 'omitnan'));
CBFSD_VSASL_1pld = squeeze(std(CBF_VSASL_1pld_est-CBF,0,1, 'omitnan'));
ATTSD_PCASL_mPLD = squeeze(std(ATT_PCASL_mPLD_est-ATT_vec,0,1, 'omitnan'));
ATTSD_VSASL_mPLD = squeeze(std(ATT_VSASL_mPLD_est-ATT_vec,0,1, 'omitnan'));
%-------accuracy: Bias -------------------
figure('Units','normalized','Position', [0, 0, 0.8, 1]);
subplot(3,2,1); hold on; grid on;
plot(ATT, CBFBIAS_PCASL_1pld(:,4)','b--','LineWidth', 3);
plot(ATT, CBFBIAS_VSASL_1pld(:,3)','g--','LineWidth',3);
plot(ATT, CBFBIAS_PCASL_mPLD,'b-','LineWidth',5); 
plot(ATT, CBFBIAS_VSASL_mPLD,'g-','LineWidth',5); 
plot(ATT, CBFBIAS,'r-','LineWidth',3); 
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
ylim([-60 20]); yticks([-60:20:20]);
xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)'); 
ylabel({'Bias', '(mL/100g/min)'}); 
set(gca,'FontSize',20, 'fontweight','bold'); box on;
% title('fitted CBF','FontSize',20,'FontWeight','bold');

subplot(3,2,2); hold on; grid on;
plot(ATT, ATTBIAS_PCASL_mPLD,'b-','LineWidth',5);
plot(ATT, ATTBIAS_VSASL_mPLD,'g-','LineWidth',5);
% plot(ATT, ATTBIAS(:,2)'./(ATT+1)*100,'m-','LineWidth',5);
plot(ATT, ATTBIAS,'r-','LineWidth',3);
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
xlim([0,4000]); xticks(0:1000:4000);
ylim([-1500 1000]);yticks(-1500:500:1000);
% xlabel('true ATT (ms)'); 
ylabel({'Bias', '(ms)'}); 
set(gca,'FontSize',20,'fontweight','bold'); box on
% title('fitted ATT','FontSize',20,'FontWeight','bold');
hold off;


%--------Precision: SD-----------------------------
subplot(3,2,3); hold on; grid on;
plot(ATT, CBFSD_PCASL_1pld(:,4), 'b--','LineWidth',3);
plot(ATT, CBFSD_VSASL_1pld(:,3), 'g--','LineWidth',3);
plot(ATT, CBFSD_PCASL_mPLD,'b-','LineWidth',5); 
plot(ATT, CBFSD_VSASL_mPLD,'g-','LineWidth',5);
% plot(ATT, CBFCoV(:,2)','m-','LineWidth',5); 
plot(ATT, CBFSD,'r-','LineWidth',3); 
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
ylim([0 40]); yticks(0:10:40);
xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)'); 
ylabel({'SD', '(mL/100g/min)'}); 
set(gca,'FontSize',20, 'FontWeight','bold'); box on
% title('fitted CBF','FontSize',20,'FontWeight','bold');
hold off;

subplot(3,2,4); hold on; grid on;
plot(ATT, ATTSD_PCASL_mPLD,'b-','LineWidth',5);
plot(ATT, ATTSD_VSASL_mPLD,'g-','LineWidth',5);
% plot(ATT, ATTCoV(:,2)'*100,'m-','LineWidth',5);
plot(ATT, ATTSD,'r-','LineWidth',3);
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
xlim([0, 4000]);xticks(0:1000:4000);
ylim([0 1500]); yticks(0:500:1500);
% xlabel('true ATT (ms)'); 
ylabel({'SD', '(ms)'}); 
set(gca,'FontSize',20,'FontWeight','bold'); box on
% title('fitted ATT','FontSize',20,'FontWeight','bold');
hold off;

%-------------------Acuracy + Precision: RMSE-----------------------------
%  Figure 3: RSME of (A) CBF  (B) ATT
subplot(3,2,5); hold on; grid on;
plot(ATT, CBFRMSE_PCASL_1pld(:,4), 'b--','LineWidth',3);
plot(ATT, CBFRMSE_VSASL_1pld(:,3), 'g--','LineWidth',3);
plot(ATT, CBFRMSE_PCASL_mPLD, 'b-','LineWidth',5);
plot(ATT, CBFRMSE_VSASL_mPLD, 'g-','LineWidth',5);
% plot(ATT, CBFRMSE(:,2)', 'm-','LineWidth',5);
plot(ATT, CBFRMSE, 'r-','LineWidth',3);
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
xlim([0,4000]);xticks(0:1000:4000);
ylim([0 80]); yticks(0:20:80);
xlabel('true ATT (ms)'); 
ylabel({'RMSE', '(mL/100g/min)'}); 
set(gca,'FontSize',20,'FontWeight','bold'); box on
% title('fitted CBF','FontSize',20,'FontWeight','bold');
% legend({'PCASL PLD=2000ms', 'VSASL PLD=1500ms', 'PCASL multi-PLD','VSASL multi-PLD','MULTIVERSE'},'FontSize',16,'Location','north');
hold off;

subplot(3,2,6); hold on; grid on;
plot(ATT, ATTRMSE_PCASL_mPLD, 'b-','LineWidth',5);
plot(ATT, ATTRMSE_VSASL_mPLD, 'g-','LineWidth',5);
% plot(ATT, ATTRMSE(:,2)', 'm-','LineWidth',5);
plot(ATT, ATTRMSE, 'r-','LineWidth',3);
plot([ATT(1),ATT(end)],[0,0],'k-','LineWidth',1.5);
xlim([0,4000]);xticks(0:1000:4000);
ylim([0 2000]); yticks(0:500:2000);
xlabel('true ATT (ms)'); 
ylabel({'RMSE', '(ms)'}); 
set(gca,'FontSize',20,'FontWeight','bold'); box on
% title('fitted ATT','FontSize',20,'FontWeight','bold');
%--------------end of absolute plot------------------------
% plot the Initial value for different ATT 
figure('Units','normalized','Position', [0, 0, 1, 0.5]);
subplot(1,2,1);
errorbar(ATT, mean(record_Init_allthree(:,:,1,2),1), std(record_Init_allthree(:,:,1,2),0, 1),'b-', 'LineWidth',2); 
hold on; grid on;
errorbar(ATT, mean(record_Init_allthree(:,:,1,3),1), std(record_Init_allthree(:,:,1,3),0, 1), 'g-', 'LineWidth',2); 
errorbar(ATT, mean(record_Init_allthree(:,:,1,1),1), std(record_Init_allthree(:,:,1,1),0, 1), 'r-', 'LineWidth',2); 
xlim([0,4000]);xticks(0:1000:4000);
ylim([0 100]); yticks(0:20:100);
xlabel('true ATT (ms)'); 
ylabel(['CBF_0', '(mL/100g/min)']); 
set(gca,'FontSize',20,'FontWeight','bold'); box on;
title('CBF initial value');

subplot(1,2,2);
errorbar(ATT, mean(record_Init_allthree(:,:,2,2),1), std(record_Init_allthree(:,:,2,2),0, 1),'b-', 'LineWidth',2);
hold on; grid on;
errorbar(ATT, mean(record_Init_allthree(:,:,2,3),1), std(record_Init_allthree(:,:,2,3),0, 1), 'g-', 'LineWidth',2); 
errorbar(ATT, mean(record_Init_allthree(:,:,2,1),1), std(record_Init_allthree(:,:,2,1),0, 1), 'r-', 'LineWidth',2); 
xlim([0,4000]);xticks(0:1000:4000);
ylim([0 5500]); yticks(0:500:5500);
xlabel('true ATT (ms)'); 
ylabel(['ATT_0', '(ms)']); 
set(gca,'FontSize',20,'FontWeight','bold'); box on;
title('ATT initial value');
legend({'PCASL multi-PLD','VSASL multi-PLD','MULTIVERSE'},'FontSize',16,'Location','northwest');

% threshold_CBF_CI = 1e100 %3e3;
% threshold_ATT_CI = 1e100 %3e6;
% % absolute CI
% figure('Units','normalized','Position', [0, 0, 1, 0.6]);box on;
% subplot(1,2,1);hold on;
% % errorbar(ATT, squeeze(mean(conintval_PCASL_mPLD(:,:,1),1,"omitnan")), squeeze(std(conintval_PCASL_mPLD(:,:,1),0,1,"omitmissing")), 'b+-', 'LineWidth',3);
% tmp=conintval_PCASL_mPLD(:,:,1);
% tmp1 = tmp> threshold_CBF_CI;
% tmp(tmp1) = NaN;
% plot(ATT, squeeze(mean(tmp,1,"omitnan")),'b+-', 'LineWidth',3);
% tmp=conintval_VSASL_mPLD(:,:,1);
% tmp1 = tmp> threshold_CBF_CI;
% tmp(tmp1) = NaN;
% % errorbar(ATT, squeeze(mean(tmp,1,"omitnan")), squeeze(std(tmp,0,1,"omitmissing")), 'g+-', 'LineWidth',3);  
% plot(ATT, squeeze(mean(tmp,1,"omitnan")),'g+-', 'LineWidth',3);
% tmp = conintval_pcvs_est(:,:,1,1);
% tmp1 = tmp> threshold_CBF_CI;
% tmp(tmp1) = NaN;
% % errorbar(ATT, squeeze(mean(tmp,1,"omitnan")), squeeze(std(conintval_pcvs_est(:,:,1,6),0,1,"omitmissing")), 'r+-', 'LineWidth',3);
% plot(ATT, squeeze(mean(tmp,1,"omitnan")),'r+-', 'LineWidth',3);
% xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)','FontSize',20); ylabel('CI (ml/100g/min)','FontSize',20); title('fitted CBF');
% legend({'Multi-PLD PCASL', 'Multi-PLD VSASL','Multi-PLD MULTIVERSE'},'FontSize',18,'Location','northeast');
% set(gca,'FontSize',20, 'fontweight','bold'); 
% 
% subplot(1,2,2);hold on;
% % errorbar(ATT, squeeze(mean(conintval_PCASL_mPLD(:,:,2),1,"omitnan")), squeeze(std(conintval_PCASL_mPLD(:,:,2),0,1,"omitmissing")), 'b+-', 'LineWidth',3);
% tmp=conintval_PCASL_mPLD(:,:,2);
% tmp1 = tmp> threshold_ATT_CI;
% tmp(tmp1) = NaN;
% plot(ATT, squeeze(mean(tmp,1,"omitnan")),'b+-', 'LineWidth',3);
% tmp=conintval_VSASL_mPLD(:,:,2);
% tmp1 = tmp> threshold_ATT_CI;
% tmp(tmp1) = NaN;
% % errorbar(ATT, squeeze(mean(tmp,1,"omitnan")), squeeze(std(tmp,0,1,"omitmissing")), 'g+-', 'LineWidth',3);  
% plot(ATT, squeeze(mean(tmp,1,"omitnan")),'g+-', 'LineWidth',3);
% tmp=conintval_pcvs_est(:,:,2,6);
% tmp1 = tmp> threshold_ATT_CI;
% tmp(tmp1) = NaN;
% % errorbar(ATT, squeeze(mean(conintval_pcvs_est(:,:,2,6),1,"omitnan")), squeeze(std(conintval_pcvs_est(:,:,2,6),0,1,"omitmissing")), 'r+-', 'LineWidth',3);
% plot(ATT, squeeze(mean(tmp,1,"omitnan")),'r+-', 'LineWidth',3);
% xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)','FontSize',18); ylabel('CI (ms)','FontSize',20); title('fitted ATT');
% set(gca,'FontSize',20, 'fontweight','bold'); 
% 
% % normalized CI
% figure('Units','normalized','Position', [0, 0, 1, 0.6]);box on;
% subplot(1,2,1);hold on;
% tmp=conintval_PCASL_mPLD(:,:,1);
% tmp1 = tmp> threshold_CBF_CI;
% tmp(tmp1) = NaN;
% % errorbar(ATT, squeeze(mean(conintval_PCASL_mPLD(:,:,1)./CBF_PCASL_mPLD_est,1,"omitnan")), squeeze(std(conintval_PCASL_mPLD(:,:,1)./CBF_PCASL_mPLD_est,0,1,"omitmissing")), 'b+-', 'LineWidth',3);
% plot(ATT, squeeze(mean(tmp./CBF_PCASL_mPLD_est,1,"omitnan")),'b+-', 'LineWidth',3);
% tmp=conintval_VSASL_mPLD(:,:,1);
% tmp1 = tmp> threshold_CBF_CI;
% tmp(tmp1) = NaN;
% % errorbar(ATT, squeeze(mean(tmp./CBF_VSASL_mPLD_est,1,"omitnan")), squeeze(std(tmp./CBF_VSASL_mPLD_est,0,1,"omitmissing")), 'g+-', 'LineWidth',3);  
% plot(ATT, squeeze(mean(tmp./CBF_VSASL_mPLD_est,1,"omitnan")),'g+-', 'LineWidth',3);
% % errorbar(ATT, squeeze(mean(conintval_pcvs_est(:,:,1,6)./CBF_pcvs_est(:,:,6),1,"omitnan")), squeeze(std(conintval_pcvs_est(:,:,1,6)./CBF_pcvs_est(:,:,6),0,1,"omitmissing")), 'r+-', 'LineWidth',3);
% tmp=conintval_pcvs_est(:,:,1,6);
% tmp1 = tmp> threshold_CBF_CI;
% tmp(tmp1) = NaN;
% plot(ATT, squeeze(mean(tmp./CBF_pcvs_est(:,:,6),1,"omitnan")), 'r+-', 'LineWidth',3);
% xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)','FontSize',20); ylabel('nCI ','FontSize',20); title('fitted CBF');
% legend({'Multi-PLD PCASL', 'Multi-PLD VSASL','Multi-PLD MULTIVERSE'},'FontSize',18,'Location','northeast');
% set(gca,'FontSize',20, 'fontweight','bold'); 
% 
% subplot(1,2,2);hold on;
% tmp=conintval_PCASL_mPLD(:,:,2);
% tmp1 = tmp>threshold_ATT_CI;
% tmp(tmp1) = NaN;
% % errorbar(ATT, squeeze(mean(conintval_PCASL_mPLD(:,:,2)./ATT_PCASL_mPLD_est,1,"omitnan")), squeeze(std(conintval_PCASL_mPLD(:,:,2)./ATT_PCASL_mPLD_est,0,1,"omitmissing")), 'b+-', 'LineWidth',3);
% plot(ATT, squeeze(mean(tmp./ATT_PCASL_mPLD_est,1,"omitnan")), 'b+-', 'LineWidth',3);
% tmp=conintval_VSASL_mPLD(:,:,2);
% tmp1 = tmp>threshold_ATT_CI;
% tmp(tmp1) = NaN;
% % errorbar(ATT, squeeze(mean(tmp./ATT_VSASL_mPLD_est,1,"omitnan")), squeeze(std(tmp./ATT_VSASL_mPLD_est,0,1,"omitmissing")), 'g+-', 'LineWidth',3);  
% plot(ATT, squeeze(mean(tmp./ATT_VSASL_mPLD_est,1,"omitnan")),'g+-', 'LineWidth',3);  
% % errorbar(ATT, squeeze(mean(conintval_pcvs_est(:,:,2,6)./ATT_pcvs_est(:,:,6),1,"omitnan")), squeeze(std(conintval_pcvs_est(:,:,2,6)./ATT_pcvs_est(:,:,6),0,1,"omitmissing")), 'r+-', 'LineWidth',3);
% tmp = conintval_pcvs_est(:,:,2,6);
% tmp1 = tmp>threshold_ATT_CI;
% tmp(tmp1) = NaN;
% plot(ATT, squeeze(mean(tmp./ATT_pcvs_est(:,:,6),1,"omitnan")), 'r+-', 'LineWidth',3);
% xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)','FontSize',18); ylabel('nCI ','FontSize',20); title('fitted ATT');
% set(gca,'FontSize',20, 'fontweight','bold'); 
% 
% % R2 fit, R2 adjusted
% figure('Units','normalized','Position', [0, 0, 1, 0.6]);box on;
% subplot(1,2,1);hold on;
% plot(ATT, squeeze(mean(R2fit_PCASL_mPLD,1,"omitnan")),'b+-', 'LineWidth',3);
% plot(ATT, squeeze(mean(R2fit_VSASL_mPLD,1,"omitnan")),'g+-', 'LineWidth',3);
% plot(ATT, squeeze(mean(R2fit_pcvs(:,:,6),1,"omitnan")),'r+-', 'LineWidth',3);
% xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)','FontSize',20); ylabel('R2','FontSize',20); title('fitted CBF');
% legend({'Multi-PLD PCASL', 'Multi-PLD VSASL','Multi-PLD MULTIVERSE'},'FontSize',18,'Location','northeast');
% set(gca,'FontSize',20, 'fontweight','bold'); 
% 
% subplot(1,2,2);hold on;
% plot(ATT, squeeze(mean(R2adj_PCASL_mPLD,1,"omitnan")),'b+-', 'LineWidth',3);
% plot(ATT, squeeze(mean(R2adj_VSASL_mPLD,1,"omitnan")),'g+-', 'LineWidth',3);
% plot(ATT, squeeze(mean(R2adj_pcvs(:,:,6),1,"omitnan")),'r+-', 'LineWidth',3);
% xlim([0,4000]);xticks(0:1000:4000);
% xlabel('true ATT (ms)','FontSize',20); ylabel('R2 adjusted','FontSize',20); title('fitted CBF');
% set(gca,'FontSize',20, 'fontweight','bold'); 

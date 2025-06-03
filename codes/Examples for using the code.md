# Examples for using the code

## 1 generate VSASL signal

```matlab
%----code start----

CBF = 60;

cbf = CBF/6000;

T1_app = 1600;

T1_artery = 1850;

T_tau = 1800; % PCASL labeling duration

T2_factor=1; % no crusher effect; crusher effect 0.78

alpha_BS1=1; % no background suppression effect; BS effect 0.95

lambda = 0.9;

alpha_VSASL = 0.56;

M0_b=1;

True_ATT = 1600;

myPLDs = 500:500:3000;  % this is Matlab reference data, X

% [cbf;True_ATT] % this fitted beta or [a;b]

true_vsasl_curve = fun_VSASL_1comp_vect_pep([cbf;True_ATT], myPLDs, T1_artery, T2_factor, alpha_BS1,alpha_VSASL);

true_vsasl_curve= [ 0.0047    0.0072    0.0083    0.0068    0.0052    0.0039]; % this is Matlab reference data, Y

figure; plot(myPLDs, true_vsasl_curve);

[beta, conintval] = fit_VSASL_vect_pep(myPLDs', true_vsasl_curve,T1_artery, T2_factor, alpha_BS1,alpha_VSASL);

fit_cbf = beta(1)*6000;

fit_ATT = beta(2); 

fitted_vsasl_curve = fun_VSASL_1comp_vect_pep(beta, myPLDs, T1_artery, T2_factor, alpha_BS1,alpha_VSASL);

figure; plot(myPLDs, fitted_vsasl_curve, ‘r’);
```

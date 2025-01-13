1/2/2025
Feng:  I am thinking of following Chen Lin et al’s method (DOI: 10.1038/s41467-020-14874-0  full paper is in the papers folder) to train the network with simulation data and test it on the simulation and real data. They are fitting some other model for CEST imaging. We are fitting perfusion model using arterial spin labeling MRI method.
Code: MCsimu_PCpVSASL_multiPLD_onePLD_rmse_scalNoise_20241226.m is for the process of simulating the ASL data with noise and perform the fitting using three methods: (1) PCASL (2) VSASL and (3) MULTIVERSE (PC+VS) ASL. The first two are previously established methods. The (3) MULTIVERSE is the new framework that leverages the previous two methods to improve the cerebral blood flow (CBF, or perfusion) and arterial transit time (ATT) estimation. This work was presented at ISMRM 2024 and you can read the abstract. I also provide a brief instruction about each methods’ model in KineticEquationsForASL.doc. You do not need to understand the math, just need to know which function is from which code is fine.
 
1/6/2025
Feng: The first step is to transfer the following Matlab codes into Python codes.
fun_PCASL_1comp_vect_pep
fun_VSASL_1comp_vect_pep
fit_PCASL_vectInit_pep
fit_ fit_VSASL_vectInit_pep
 
check out the Examples for using the code.docx in the codes folder for how to use those functions, just make python function use like that.
 
1/10/2025
Feng: translate fun_VSASL_1comp_vect_pep.m And fit_VSASL_vect_nopep.m  to Python first, once figure out how to pass external paramters, then translate fit_VSASL_vect_pep.m
 
Matlab reference data:  please see Examples for using the code.doc in the code folder.
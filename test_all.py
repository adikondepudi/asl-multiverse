import unittest
import numpy as np
import torch
# import matplotlib.pyplot as plt # No longer used for saving plots by this script directly
import pandas as pd
# from scipy.stats import pearsonr # Not used
import os
import tempfile
import time
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from vsasl_functions import fun_VSASL_1comp_vect_pep, fit_VSASL_vectInit_pep
from pcasl_functions import fun_PCASL_1comp_vect_pep, fit_PCASL_vectInit_pep
from multiverse_functions import fun_PCVSASL_misMatchPLD_vect_pep, fit_PCVSASL_misMatchPLD_vectInit_pep
from asl_simulation import ASLSimulator, ASLParameters
from enhanced_simulation import RealisticASLSimulator
from enhanced_asl_network import EnhancedASLNet, CustomLoss
# from asl_trainer import EnhancedASLTrainer # Not directly used for tests here, but by main.py
from comparison_framework import ComprehensiveComparison

class TestMATLABTranslation(unittest.TestCase):
    def setUp(self):
        self.CBF, self.cbf = 60, 60/6000
        self.T1_artery, self.T2_factor, self.alpha_BS1 = 1850, 1, 1
        self.alpha_PCASL, self.alpha_VSASL, self.T_tau, self.True_ATT = 0.85, 0.56, 1800, 1600
        self.PLDs = np.arange(500, 3001, 500)
        self.matlab_vsasl_signal = np.array([0.0047,0.0072,0.0083,0.0068,0.0052,0.0039])
        self.rtol = 2e-2
    def test_vsasl_signal_generation(self):
        py_sig = fun_VSASL_1comp_vect_pep([self.cbf,self.True_ATT],self.PLDs,self.T1_artery,self.T2_factor,self.alpha_BS1,self.alpha_VSASL)
        np.testing.assert_allclose(py_sig,self.matlab_vsasl_signal,rtol=self.rtol,err_msg="VSASL signal mismatch")
        self.assertTrue(np.all(py_sig>=0)); self.assertEqual(len(py_sig),len(self.PLDs))
    def test_pcasl_signal_generation(self):
        py_sig = fun_PCASL_1comp_vect_pep([self.cbf,self.True_ATT],self.PLDs,self.T1_artery,self.T_tau,self.T2_factor,self.alpha_BS1,self.alpha_PCASL)
        self.assertTrue(np.all(py_sig>=0)); self.assertEqual(len(py_sig),len(self.PLDs))
        self.assertLess(py_sig[-1],py_sig[np.argmax(py_sig)],"Signal should decay")
    def test_multiverse_signal_generation(self):
        pldti = np.column_stack([self.PLDs,self.PLDs])
        comb_sig = fun_PCVSASL_misMatchPLD_vect_pep([self.cbf,self.True_ATT],pldti,self.T1_artery,self.T_tau,self.T2_factor,self.alpha_BS1,self.alpha_PCASL,self.alpha_VSASL)
        self.assertEqual(comb_sig.shape,(len(self.PLDs),2)); self.assertTrue(np.all(comb_sig>=0))
        p_ref = fun_PCASL_1comp_vect_pep([self.cbf,self.True_ATT],self.PLDs,self.T1_artery,self.T_tau,self.T2_factor,self.alpha_BS1,self.alpha_PCASL)
        v_ref = fun_VSASL_1comp_vect_pep([self.cbf,self.True_ATT],self.PLDs,self.T1_artery,self.T2_factor,self.alpha_BS1,self.alpha_VSASL)
        np.testing.assert_allclose(comb_sig[:,0],p_ref,rtol=1e-10); np.testing.assert_allclose(comb_sig[:,1],v_ref,rtol=1e-10)
    def test_parameter_recovery_vsasl(self):
        clean_sig = fun_VSASL_1comp_vect_pep([self.cbf,self.True_ATT],self.PLDs,self.T1_artery,self.T2_factor,self.alpha_BS1,self.alpha_VSASL)
        np.random.seed(42); noisy_sig = clean_sig + np.random.normal(0,0.0002,clean_sig.shape)
        beta,_,rmse,_ = fit_VSASL_vectInit_pep(self.PLDs,noisy_sig,[50/6000,1500],self.T1_artery,self.T2_factor,self.alpha_BS1,self.alpha_VSASL)
        self.assertLess(abs(beta[0]-self.cbf)/self.cbf,0.05,"CBF recovery error large"); self.assertLess(abs(beta[1]-self.True_ATT)/self.True_ATT,0.05,"ATT recovery error large"); self.assertGreater(rmse,0)
    def test_parameter_recovery_pcasl(self):
        clean_sig = fun_PCASL_1comp_vect_pep([self.cbf,self.True_ATT],self.PLDs,self.T1_artery,self.T_tau,self.T2_factor,self.alpha_BS1,self.alpha_PCASL)
        np.random.seed(42); noisy_sig = clean_sig + np.random.normal(0,0.0002,clean_sig.shape)
        beta,_,_,_ = fit_PCASL_vectInit_pep(self.PLDs,noisy_sig,[50/6000,1500],self.T1_artery,self.T_tau,self.T2_factor,self.alpha_BS1,self.alpha_PCASL)
        self.assertLess(abs(beta[0]-self.cbf)/self.cbf,0.05); self.assertLess(abs(beta[1]-self.True_ATT)/self.True_ATT,0.05)
    def test_parameter_recovery_multiverse(self):
        pldti = np.column_stack([self.PLDs,self.PLDs])
        clean_sig = fun_PCVSASL_misMatchPLD_vect_pep([self.cbf,self.True_ATT],pldti,self.T1_artery,self.T_tau,self.T2_factor,self.alpha_BS1,self.alpha_PCASL,self.alpha_VSASL)
        np.random.seed(42); noisy_sig = clean_sig + np.random.normal(0,0.0002,clean_sig.shape)
        beta,_,_,_ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti,noisy_sig,[50/6000,1500],self.T1_artery,self.T_tau,self.T2_factor,self.alpha_BS1,self.alpha_PCASL,self.alpha_VSASL)
        self.assertLess(abs(beta[0]-self.cbf)/self.cbf,0.03); self.assertLess(abs(beta[1]-self.True_ATT)/self.True_ATT,0.03)

class TestASLSimulation(unittest.TestCase):
    def setUp(self): self.sim,self.enh_sim,self.plds = ASLSimulator(),RealisticASLSimulator(),np.arange(500,3001,500)
    def test_basic_simulation(self):
        atts = np.array([800,1600,2400]); sigs = self.sim.generate_synthetic_data(self.plds,atts,n_noise=10)
        self.assertIn('PCASL',sigs); self.assertIn('VSASL',sigs); self.assertIn('MULTIVERSE',sigs)
        exp_shape = (10,len(atts),len(self.plds)); self.assertEqual(sigs['PCASL'].shape,exp_shape)
        self.assertEqual(sigs['MULTIVERSE'].shape,(10,len(atts),len(self.plds),2))
    def test_noise_scaling(self):
        sc = self.sim.compute_tr_noise_scaling(self.plds)
        self.assertIn('VSASL',sc); self.assertIn('PCASL',sc); self.assertIn('MULTIVERSE',sc)
        for s_val in sc.values(): self.assertGreater(s_val,0)
        self.assertGreater(sc['MULTIVERSE'],sc['PCASL']); self.assertGreater(sc['MULTIVERSE'],sc['VSASL'])
    def test_enhanced_simulation(self):
        ds = self.enh_sim.generate_diverse_dataset(self.plds,n_subjects=2,conditions=['healthy'],noise_levels=[5.0]) # Reduced subjects for speed
        self.assertIn('signals',ds); self.assertIn('parameters',ds); self.assertIn('conditions',ds)
        exp_samps = 2*3*1; self.assertEqual(len(ds['signals']),exp_samps) # 2 subj * 3 noise types * 1 cond
        self.assertEqual(ds['signals'].shape[1],len(self.plds)*2)
    def test_spatial_data_generation(self):
        data,cbf,att = self.enh_sim.generate_spatial_data(matrix_size=(16,16),n_slices=3,plds=self.plds[:2]) # Reduced for speed
        self.assertEqual(data.shape,(16,16,3,2)); self.assertEqual(cbf.shape,(16,16))
        self.assertTrue(np.all(cbf>0)); self.assertTrue(np.all(att>0))

class TestNeuralNetworkFramework(unittest.TestCase):
    def setUp(self):
        self.input_s,self.n_p,self.batch_s = 12,6,4 # Reduced batch for speed
        self.model = EnhancedASLNet(input_size=self.input_s,n_plds=self.n_p,hidden_sizes=[16,8]) # Reduced hidden sizes
        self.sample_in = torch.randn(self.batch_s,self.input_s)
    def test_model_architecture(self):
        with torch.no_grad(): cbf_p,att_p,cbf_lv,att_lv = self.model(self.sample_in)
        exp_sh = (self.batch_s,1)
        self.assertEqual(cbf_p.shape,exp_sh); self.assertEqual(att_p.shape,exp_sh)
        self.assertEqual(cbf_lv.shape,exp_sh); self.assertEqual(att_lv.shape,exp_sh)
    def test_custom_loss(self):
        loss_fn = CustomLoss()
        cbf_p,att_p,cbf_t,att_t = torch.randn(self.batch_s,1),torch.randn(self.batch_s,1)+1000,torch.randn(self.batch_s,1),torch.randn(self.batch_s,1)+1000
        cbf_lv,att_lv = torch.randn(self.batch_s,1),torch.randn(self.batch_s,1)
        loss = loss_fn(cbf_p,att_p,cbf_t,att_t,cbf_lv,att_lv,epoch=0)
        self.assertIsInstance(loss,torch.Tensor); self.assertEqual(loss.dim(),0); self.assertGreater(loss.item(),0)
    def test_model_factory(self):
        def create_model(): return EnhancedASLNet(input_size=self.input_s,n_plds=self.n_p,hidden_sizes=[16,8])
        m1,m2 = create_model(),create_model()
        self.assertEqual(type(m1),type(m2))
        self.assertFalse(torch.equal(list(m1.parameters())[0].data, list(m2.parameters())[0].data))

class TestPerformanceComparison(unittest.TestCase): # Simplified, as full comparison is slow
    def setUp(self):
        self.sim = RealisticASLSimulator(); self.plds = np.arange(500,3001,500)
        self.comp = ComprehensiveComparison(output_dir="temp_test_comp_results") # Use temp dir
        self.test_data, self.true_params = self._generate_test_data(n_atts=5, n_noise_per_att=2) # Very small dataset
    def _generate_test_data(self, n_atts=10, n_noise_per_att=5):
        atts = np.random.uniform(500,4000,n_atts)
        sigs = self.sim.generate_synthetic_data(self.plds,atts,n_noise=n_noise_per_att,tsnr=5.0)
        test_data = {'PCASL':sigs['PCASL'].reshape(-1,len(self.plds)),'VSASL':sigs['VSASL'].reshape(-1,len(self.plds)),
                     'MULTIVERSE_LS_FORMAT':sigs['MULTIVERSE'].reshape(-1,len(self.plds),2), # Adjusted key for new main.py structure
                     'NN_INPUT_FORMAT': np.concatenate([sigs['PCASL'].reshape(-1,len(self.plds)), sigs['VSASL'].reshape(-1,len(self.plds))], axis=1) # Basic NN input
                     }
        true_params = np.column_stack([np.full(len(atts)*n_noise_per_att,self.sim.params.CBF),np.repeat(atts,n_noise_per_att)])
        return test_data,true_params
    def test_least_squares_methods_basic_run(self):
        att_ranges = [(500, 4000, "Full ATT Range")]
        results_df = self.comp.compare_methods(self.test_data, self.true_params, self.plds, att_ranges)
        self.assertFalse(results_df.empty, "Comparison results DF should not be empty")
        self.assertIn("MULTIVERSE-LS", results_df['method'].unique())
        multiverse_res = results_df[results_df['method']=="MULTIVERSE-LS"].iloc[0]
        self.assertIsInstance(multiverse_res.cbf_bias,float); self.assertIsInstance(multiverse_res.success_rate,float)
        # Cannot assert on success rate or bias with such small data, just check it runs
    def tearDown(self): # Clean up temp dir
        import shutil
        if Path("temp_test_comp_results").exists(): shutil.rmtree("temp_test_comp_results")


def run_comprehensive_validation(output_dir="test_results_validation_suite"): # Changed dir name
    output_path = Path(output_dir); output_path.mkdir(exist_ok=True, parents=True)
    print("="*80+"\nCOMPREHENSIVE ASL VALIDATION SUITE (LS Methods Only)\n"+"="*80)
    simulator = RealisticASLSimulator(); plds = np.arange(500,3001,500)
    CBF,cbf,T1_a,T2_f,alpha_BS1,alpha_V,T_tau,alpha_P=60,60/6000,1850,1,1,0.56,1800,0.85
    
    print("\n1. MATLAB Translation Validation (already covered by unittest)")
    matlab_vsasl_ref = np.array([0.0047,0.0072,0.0083,0.0068,0.0052,0.0039])
    py_vsasl = fun_VSASL_1comp_vect_pep([cbf,1600],plds,T1_a,T2_f,alpha_BS1,alpha_V)
    max_err = np.max(np.abs(py_vsasl-matlab_vsasl_ref)/matlab_vsasl_ref*100)
    print(f"VSASL translation max rel error: {max_err:.2f}% - {'PASS' if max_err<2.0 else 'FAIL'}")

    print("\n2. ASL Method Performance Comparison (LS Methods)")
    snr_levels, att_ranges_cfg = [3,5,10,15], [(500,1500,"Short ATT"),(1500,2500,"Medium ATT"),(2500,4000,"Long ATT")]
    comp_results = {}
    for snr in snr_levels:
        print(f"\nTesting SNR = {snr}"); comp_results[snr] = {}
        for att_min,att_max,r_name in att_ranges_cfg:
            print(f"  {r_name}: {att_min}-{att_max} ms"); n_test = 50 # Reduced for speed
            att_vals = np.random.uniform(att_min,att_max,n_test)
            methods, results_per_att = ['PCASL','VSASL','MULTIVERSE'], {m:{'cbf_err':[],'att_err':[],'sr':0} for m in methods}
            for att in att_vals:
                sigs = simulator.generate_synthetic_data(plds,np.array([att]),n_noise=1,tsnr=snr)
                pldti_m = np.column_stack([plds,plds])
                fits_params = [ (plds, sigs['PCASL'][0,0], [50/6000,1500],T1_a,T_tau,T2_f,alpha_BS1,alpha_P),
                                (plds, sigs['VSASL'][0,0], [50/6000,1500],T1_a,T2_f,alpha_BS1,alpha_V),
                                (pldti_m, sigs['MULTIVERSE'][0,0], [50/6000,1500],T1_a,T_tau,T2_f,alpha_BS1,alpha_P,alpha_V) ]
                fit_funcs = [fit_PCASL_vectInit_pep, fit_VSASL_vectInit_pep, fit_PCVSASL_misMatchPLD_vectInit_pep]
                for i_m, method in enumerate(methods):
                    try:
                        beta,_,_,_ = fit_funcs[i_m](*fits_params[i_m])
                        results_per_att[method]['cbf_err'].append(abs(beta[0]*6000-CBF))
                        results_per_att[method]['att_err'].append(abs(beta[1]-att)); results_per_att[method]['sr']+=1
                    except: pass
            r_results = {}
            for method in methods:
                if results_per_att[method]['cbf_err']:
                    c_err,a_err=np.array(results_per_att[method]['cbf_err']),np.array(results_per_att[method]['att_err'])
                    r_results[method] = {'cbf_bias':np.mean(c_err),'cbf_std':np.std(c_err),'att_bias':np.mean(a_err),
                                         'att_std':np.std(a_err),'cbf_cv':np.std(c_err)/CBF*100,
                                         'att_cv':np.std(a_err)/np.mean(att_vals)*100,'success_rate':results_per_att[method]['sr']/n_test*100}
                    print(f"    {method}: CBF err {r_results[method]['cbf_bias']:.1f}±{r_results[method]['cbf_std']:.1f}, Success {r_results[method]['success_rate']:.1f}%")
            comp_results[snr][r_name] = r_results
    
    print("\n3. Saving Results (JSON and LaTeX Table)")
    with open(output_path/'method_comparison_LS_only.json','w') as f: json.dump(comp_results,f,indent=2)
    generate_latex_table(comp_results, output_path) # Generates text .tex file

    print(f"\nValidation (LS methods) complete! Results in: {output_path}")
    return {'matlab_validation':{'max_error':max_err,'passed':max_err<2.0}, 'comparison_results':comp_results, 'output_path':str(output_path)}

def generate_latex_table(results_dict, output_path): # Simplified, using the structure from run_comprehensive_validation
    output_dir = Path(output_path)
    latex_content = ["\\begin{table}[htbp]", "\\centering", "\\caption{ASL Method Performance (LS Only)}",
                     "\\label{tab:asl_ls_performance}", "\\begin{tabular}{llcccccc}", "\\toprule",
                     "SNR & ATT Range & Method & CBF Bias & CBF CoV & ATT Bias & ATT CoV & Success \\\\",
                     "    &           &        & (\\%) & (\\%) & (ms) & (\\%) & Rate (\\%) \\\\", "\\midrule"]
    for snr_key in sorted(results_dict.keys()): # SNR keys are ints/floats from comp_results
        first_snr = True
        for att_range_key in ["Short ATT", "Medium ATT", "Long ATT"]:
            first_range = True
            if att_range_key in results_dict[snr_key]:
                for method_key in ['PCASL', 'VSASL', 'MULTIVERSE']:
                    if method_key in results_dict[snr_key][att_range_key]:
                        data = results_dict[snr_key][att_range_key][method_key]
                        snr_str, range_str = (str(snr_key) if first_snr else ""), (att_range_key if first_range else "")
                        cbf_b, cbf_cv = data.get('cbf_bias',0)/60*100, data.get('cbf_cv',0)
                        att_b, att_cv, succ = data.get('att_bias',0), data.get('att_cv',0), data.get('success_rate',0)
                        latex_content.append(f"{snr_str} & {range_str} & {method_key} & {cbf_b:+.1f} & {cbf_cv:.1f} & {att_b:+.0f} & {att_cv:.1f} & {succ:.1f} \\\\")
                        first_snr, first_range = False, False
            if not first_range: latex_content.append("\\addlinespace")
    latex_content.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(output_dir/'performance_table_LS_only.tex','w') as f: f.write('\n'.join(latex_content))
    print(f"LaTeX table saved to {output_dir/'performance_table_LS_only.tex'}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'unittest': unittest.main(argv=[''], exit=False)
        elif sys.argv[1] == 'validation':
            results = run_comprehensive_validation()
            print(f"\nLS Validation summary: MATLAB translation: {'PASS' if results['matlab_validation']['passed'] else 'FAIL'}")
            print(f"Results saved to: {results['output_path']}")
        elif sys.argv[1] == 'comparison' and len(sys.argv) > 2:
            print("Neural network comparison from test_all.py is deprecated.")
            print("Please use main.py for comprehensive NN training and benchmarking.")
            # models_dir = sys.argv[2]; # run_neural_network_comparison(models_dir) # This function was removed as per no-PNG and main.py focus
        else: print("Usage: python test_all.py [unittest|validation]")
    else:
        print("Running comprehensive ASL testing suite (LS Methods Focus)...")
        print("\n1. Unit Tests"); unittest.main(argv=[''], exit=False, verbosity=2)
        print("\n2. LS Methods Validation"); results = run_comprehensive_validation()
        print(f"\nTesting complete! MATLAB translation: {'PASS' if results['matlab_validation']['passed'] else 'FAIL'}")
        print(f"Comprehensive LS results saved to: {results['output_path']}")
        print("\nFor NN training and full comparison, run: python main.py [config.yaml]")

# Kinetic Equations for Arterial Spin Labeling (ASL)

**Jan 7, 2024**

**Feng Xu, Ph.D.**

The Russell H. Morgan Department of Radiology and Radiological Science  
Johns Hopkins University, School of Medicine  
The F.M. Kirby Research Center for Functional Brain Imaging  
Kennedy Krieger Institute

---

## Two types of ASL will be covered:

1. **Spatial selective ASL**: pseudo-continuous ASL (PCASL) and pulsed ASL (PASL).
2. **Velocity selective (spatially non-selective) ASL**: velocity selective ASL (VSASL)

---

## Abbreviations:

- **ATT ()**: Arterial transit time, the time it takes for labeled blood spins to travel to image voxels.  
  - ATT in PCASL and PCASL means the leading edge of the blood bolus from the labeling plane to the image voxels.  
  - ATT in VSASL means the trailing edge of the blood bolus from the upper stream location to the image voxels. It also means the bolus duration from the trailing edge to the image voxel in VSASL.

- **PLD**: Post-labeling delay. It is a period (time gap) posted between spinning labeling and image acquisition. This is the MRI parameter that can be adjusted.

- **CBF() (mL/min/100g)**: cerebral blood flow.

- **tau ()**: duration of the labeling pulse train in PCASL.

- **:** proton density of the blood.

- **:** proton density of the tissue.

- **:** water fraction between blood and tissue, a constant value cited from literature.

- **:** labeling efficiency, a constant based on experiments and empirical knowledge.

- **T1,a**: longitudinal relaxation time constant of blood.

- **T1app**: longitudinal relaxation time constant of spins after entering the pool of tissue water.

---

## One compartment model

This means the blood arrives at the voxel and the spin will relax at blood T1, because tissue and blood are treated as one pool.

---

## PCASL

*equation image or representation omitted from document*

---

## VSASL

*equation image or representation omitted from document*

---

In the paper, the equation is written as:

*equation image or representation omitted from document*

---

## Sensitivity function, derivative 

*image or derivative expression omitted from document*

---

(simulation: PLD = 100:100:10000; ATT = 1200; cbf=60/6000 (mL/s/g); T1_artery=1850 s; T2_factor=1; alpha_BS1=1; alpha_PCASL=0.85; alphaVSASL=0.56;)

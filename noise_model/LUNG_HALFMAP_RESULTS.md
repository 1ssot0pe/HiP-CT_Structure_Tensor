# Lung half-map analysis (LADAF-2022-13, 16.547 um)

```
=== LUNG HALF-MAP ANALYSIS (LADAF-2022-13 lung-left, 16.547 um) ===
cubic block (80, 80, 80); A mean=-0.0004 B mean=-0.0004
noise n std=0.00030; averaged-map std=0.00345; frac avg==0 0.0%
------------------------------------------------------------------
[T1] pure-noise n: std=0.000302 (half-map level), full-recon sigma_n~0.000213
[T1] high-sigma slope=-3.37 -> CORRELATED (slope -3.4 != -5 -> low-frequency); knee ell=0.70 vox
[T1] N is ANISOTROPIC (lambda1/lambda3 @sigma=4 = 1.69); principal axis @sigma=4 = [ 0.07  0.29 -0.96] (check vs rotation axis for ring bias)
[T1] saved halfmap_T1_noise.png
------------------------------------------------------------------
[T2] c_s>thr on avg @sigma=4.0: uncorrected 14% -> MEASURED-N corrected 13% (clip 0%)  => c_s SURVIVES -> complexity is largely REAL tissue
[T2] saved halfmap_T2_premise.png
------------------------------------------------------------------
[T3] orientation-FSC optimum: sigma=1, rho=1 (cutoff 0.0083 cyc/vox)
[T3] saved halfmap_T3_certify.png
------------------------------------------------------------------
[T4] image FSC cutoff = 0.4950 cyc/vox -> resolution ~2.02 vox (~33 um)
[T4] saved halfmap_T4_resolution.png
==================================================================
```

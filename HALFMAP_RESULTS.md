# Half-map analysis results

```
=== SYNTHETIC SELF-TEST (phantom + 2 noise realisations) ===
volume (96, 96, 96), A mean=0.343 B mean=0.343
------------------------------------------------------------------
[T1] pure-noise n: std=0.25 (half-map level), full-recon sigma_n~0.177
[T1] high-sigma slope=-5.14 -> WHITE (slope ~ -5); knee ell=0.70 vox
[T1] N is ~isotropic (lambda1/lambda3 @sigma=4 = 1.15); principal axis @sigma=4 = [-0.66 -0.13 -0.74] (check vs rotation axis for ring bias)
[T1 self-test] injected white sigma_n=0.25: trN(4) meas=2.028e-06 vs analytic 2.055e-06 (ratio 0.99)
[T1] saved halfmap_T1_noise.png
------------------------------------------------------------------
[T2] c_s>thr on avg @sigma=4.0: uncorrected 100% -> MEASURED-N corrected 0% (clip 87%)  => c_s COLLAPSES -> complexity was largely NOISE
[T2] saved halfmap_T2_premise.png
------------------------------------------------------------------
[T3] orientation-FSC optimum: sigma=1, rho=2 (cutoff 0.4917 cyc/vox)
[T3 self-test] angular argmin sigma=1,rho=3; sep=1 cell(s), corr=-0.63 -> certifier VALID
[T3] saved halfmap_T3_certify.png
------------------------------------------------------------------
[T4] image FSC cutoff = 0.0150 cyc/vox -> resolution ~66.82 vox (~1336 um)
[T4] saved halfmap_T4_resolution.png
------------------------------------------------------------------
SUMMARY: certified sigma~1, rho~2 (certifier VALID on self-test)
```

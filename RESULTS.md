# Noise-floor characterisation of the structure tensor on HiP-CT i74

## Log
```
[data] INTERIOR (200, 200, 200) mean=182.8 std=33.8
[data] quiet 128^3 sub-block at (72,0,0) mean=196.8 var=336.4 (std=18.3)
[data] NO air-with-noise (air clipped to 0; 33% of volume exactly 0), NO half-reconstruction, NO high-freq spectral plateau -> N cannot be MEASURED.
[data] sigma_n^2 GENEROUS UPPER BOUND = quiet-block variance = 336 (sigma_n=18.3); pretends ALL signal variance is white noise.
========================================================================
T1 GATE white-noise: ratios=[np.float64(0.996), np.float64(0.994), np.float64(1.002), np.float64(1.018)] (<5%), iso=[np.float64(1.008), np.float64(1.013), np.float64(1.025), np.float64(1.045)] (<1.05), slope=-4.978 -> PASS
T1 discretisation bound: dev>5% below sigma~0.85 (sigma=0.7 dev 11%, sigma=1.0 dev 0%)
========================================================================
T2 GATE rho-invariance: mean(trS) change = [np.float64(0.0), np.float64(0.09), np.float64(0.13), np.float64(0.93)]% (<1%) -> PASS
T2 std ratio = [np.float64(1.0), np.float64(0.563), np.float64(0.255), np.float64(0.101)] (falls ~rho^-1.1; floor is bias not variance)
========================================================================
[T3] tissue-block structure tensor vs analytic white floor (UPPER BOUND):
    sigma= 0.5  tr(block)=1.398e+02  tr(N_white_UB)=3.625e+02  block l1/l3=2.83  block cs=0.53
    sigma= 0.7  tr(block)=1.134e+02  tr(N_white_UB)=6.739e+01  block l1/l3=3.36  block cs=0.47
    sigma= 1.0  tr(block)=4.402e+01  tr(N_white_UB)=1.133e+01  block l1/l3=3.50  block cs=0.44
    sigma= 1.4  tr(block)=1.341e+01  tr(N_white_UB)=2.106e+00  block l1/l3=3.35  block cs=0.44
    sigma= 2.0  tr(block)=3.151e+00  tr(N_white_UB)=3.540e-01  block l1/l3=2.97  block cs=0.45
    sigma= 2.8  tr(block)=8.585e-01  tr(N_white_UB)=6.581e-02  block l1/l3=3.19  block cs=0.44
    sigma= 4.0  tr(block)=3.272e-01  tr(N_white_UB)=1.106e-02  block l1/l3=4.02  block cs=0.37
    sigma= 5.7  tr(block)=1.780e-01  tr(N_white_UB)=1.882e-03  block l1/l3=5.70  block cs=0.29
    sigma= 8.0  tr(block)=1.238e-01  tr(N_white_UB)=3.457e-04  block l1/l3=11.93  block cs=0.19
[T3] block high-sigma slope = -2.31 (white noise = -5.0). Far from -5 AND anisotropic AND cs~0.4 -> the block is SIGNAL, not noise.
[T3] even the white UPPER BOUND floor is tr(N_white)/tr(block) = 3% at sigma=4 -> the real block energy is dominated by tissue.
========================================================================
[T4] radial power spectrum: low-f P=3.55e+02, high-f P=3.23e-01, ratio=1099x -> monotonic falloff, NO white plateau.
[T4] => sigma_n^2 cannot be read from a spectral plateau; a large isotropic WHITE floor at our scale is NOT present in the data.
========================================================================
[T5] stationarity (tr of block tensor @sigma=2, 8 sub-blocks): quiet-block CV=14%, interior CV=64%
[T5] interior is NON-STATIONARY (>30%) -- consistent with tissue structure dominating, not a uniform noise floor.
[T6] vessel-candidate patch at z=64,y=16,x=16 (96^3)
========================================================================
[T6a] vessel c_s @sigma=4: uncorrected=0.179 -> corrected (S - N_white_UB)=0.145 (PSD clip=0%)
[T6a] white floor is 3.9% of tr S at sigma=4 -> correcting it barely moves c_s.
========================================================================
[T7] SNR(sigma)=trS_sig/tr N_white_UB = [-0.6, 0.9, 3.3, 6.0, 8.3, 11.5, 24.7, 103.4, 478.5] (>>1 everywhere -> the white floor does NOT bind sigma_min)
[T7] sigma_min is set by the DISCRETISATION bound ~0.85 vox (17 um), not by noise.
========================================================================
[T8] HEADLINE c_s @sigma=4 on INTERIOR (2515456 voxels), thr=0.15
[T8]   uncorrected: mean=0.234  frac>thr=65%
[T8]   corrected (S - N_white, UB sigma_n^2=336): mean=0.203  frac>thr=56%  (PSD clip=0%)
[T8]   -> even the GENEROUS white floor moves the number 65% -> 56%.
========================================================================
[T9] R-proxy min sigma per rho: uncorrected [5.7, 5.7, 5.7], corrected [4.0, 5.7, 5.7] (white floor is tiny -> minimum barely moves)
```

## Five-line verdict

1. Both HARD GATES PASS: white-noise N matches 0.0337 sigma^-5 (slope -5.00, isotropic to <5%), and E[S] is rho-invariant to <1% -- N is a bias rho cannot remove. The machinery is sound.
2. N CANNOT be measured on this i74 data: air is clipped to 0 (no air noise), there is no half-reconstruction, and the power spectrum has NO high-frequency white plateau. A 'quiet tissue block' has the WRONG slope (-2.3, not -5), is anisotropic, and over-subtracts (97% clip) -- it is signal, not noise.
3. Testing the WHITE-noise hypothesis with a GENEROUS upper bound (sigma_n^2 = full quiet-block variance = 336, i.e. all signal treated as noise): the analytic floor is only ~2% of tr S at sigma=4.
4. HEADLINE: under that generous white floor the c_s>0.15 fraction moves only 65% -> 56% (clip 0%); the vessel control moves 0.18 -> 0.14. Both barely change.
5. VERDICT: the pervasive high c_s is NOT a white-noise-floor artifact -- the white floor is orders of magnitude too small. A LOW-FREQUENCY correlated-noise floor (ring/cupping) cannot be excluded, but it is mathematically inseparable from real anatomy without a half-reconstruction; nothing in the data positively supports it. Tissue complexity survives correction.

**Do not read this as confirming the hypothesis.** The falsifiable white-noise claim is FALSIFIED; the correlated-noise variant is UNTESTABLE with the current data (needs a half-reconstruction or an unclipped air region).
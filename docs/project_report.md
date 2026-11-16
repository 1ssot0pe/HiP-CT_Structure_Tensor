## 200 word Abstract: 

## Introduction: 

When computing structure tensors from HiP-CT images, we encounter two mathematical parameters, sigma and rho. Sigma is the differentiation scale, applied with a derivative kernel; rho is the integration scale, applied with the smoothing kernel. 

This pipeline computes the optimal (sigma, rho) pairs for a single vector field or complex vector field including higher order structure tensors. Here, we truncate the output to maximum of two vectors, as crossings with three or more fibres are unstable in the HOST-ODF WHAT. 

## Method: 
1_sigma_rho_selection

2_host_classification

3_global_tractography

4_optimising_parameters

## Results: 
- For i74, we obtain (sigma = 1.00, rho = 3.79) from Figure_1. This is consistent across seed counts from 1000 to 5000 and for both MSE/MAE as metrics in stage3's gradient descent

## Discussion: 
Ways to extend the project: 

STAGE1
- Feature adaptive Lindeberg selections
- Learning algorithms to sweep feature sizes to reduce compute 
- Apply more of Lindeberg's algorithms in automatic scale selection 
such as blob detection

STAGE2
- Using higher degrees of ODF functions to resolve crossings with n equal to or greater than 3

STAGE3
- Kissing vs Crossing using superposition, a local version of mass conservation
- Included in an optional file, mass conservation, where you sample from the 
arm of the crossing and measure the intensity at the crossing, allows one to 
demote all false, overcalled crossings in STAGE2's HOST

STAGE4
- Feature adaptive fitting 
- Sample n regions in the brain for different features, using weighted average
across obtained pairs of (sigma, rho) from this pipeline, to compute optimised global parameters

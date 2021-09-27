# Multi-endpoint assessment based on a joint model (MEJM)
## Introduction
It is a generalized framework for accurately determining the positivity thresholds of TMB and dividing patients into multiple subgroups based on a fusion analysis of immunotherapy efficacy. The model considered the discrete tumor response endpoint in addition to the continuous time-to-event (TTE) endpoint simultaneously to optimize the division from a comprehensive perspective.
## Usage:
**Input**:  
* Excel files with separate sheets record patients' observations (containing ORR and TTE endpoints, as well as other clinical indicators) and the corresponding TMB values.  

**Output**: 
1. JM parameters with standard errors and confidence intervals.  
2. The optimal TMB thresholds for the dichotomy.  
3. The optimal TMB thresholds for the trichotomy.

**NOTE1:**
The upper & lower bounds and the dimension of particle swarm optimization step need to be adjusted according to the range of values of the actual input data. 

**NOTE2:**
The calculation criteria for the composite prognosis indexes can be adjusted according to the clinical characteristics of the cancer species to be analyzed.

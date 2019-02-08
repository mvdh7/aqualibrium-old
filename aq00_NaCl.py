from copy import deepcopy
from time import time
import aqualibrium as aq
import numpy  as np
import pytzer as pz

from scipy.optimize import minimize

# Get data with pytzer
mols,ions,T = pz.io.getmols('data/NaCl3.csv')
cf = deepcopy(pz.cfdicts.M88)

# Calculate H2O thermodynamic dissociation constant
ln_kH2O = aq.dissoc.H2O_M88(T)[0]
kH2O = np.exp(ln_kH2O)

# Add H-OH to model
mols = np.concatenate((mols,np.zeros((np.shape(mols)[0],2))), axis=1)
ions = np.concatenate((ions,np.array(['H','OH'])))

# Add some H-X and X-OH interaction terms
cf.bC['H-Cl' ] = pz.coeffs.bC_H_Cl_CMR93
cf.bC['Na-OH'] = pz.coeffs.bC_Na_OH_PP87i
cf.theta['H-Na' ] = pz.coeffs.theta_H_Na_CMR93
cf.theta['Cl-OH'] = pz.coeffs.theta_Cl_OH_HMW84
cf.psi['H-Na-Cl' ] = pz.coeffs.psi_H_Na_Cl_CMR93
cf.psi['H-Na-OH' ] = pz.coeffs.psi_H_Na_OH_HMW84
cf.psi['H-Cl-OH' ] = pz.coeffs.psi_H_Cl_OH_HMW84
cf.psi['Na-Cl-OH'] = pz.coeffs.psi_Na_Cl_OH_HMW84

# Tidy up cfdict
cf.add_zeros(ions)
cf.get_contents() # temporary bug fix on next line:
cf.ions = np.unique(np.concatenate((cf.ions,ions)))

# Define solver function
def Ksolver_H2O(mols_i,ions,T_i,cf,ln_kH2O_i,p_mH):
    
    # Convert p_mH to mH
    mH = 10**-p_mH
    
    mols_i[0][ions == 'H' ] = mH
    mols_i[0][ions == 'OH'] = 0
    
    # Get mOH from charge balance
    mOH = aq.balance.charges(mols_i,ions).ravel()
    mols_i[0][ions == 'OH'] = mOH
    
    ln_acfs = pz.model.ln_acfs(mols_i,ions,T_i,cf).ravel()
    
    ln_gH  = ln_acfs[ions == 'H' ]
    ln_gOH = ln_acfs[ions == 'OH']
    
    Keq = (ln_gH + np.log(mH) + ln_gOH + np.log(mOH) - ln_kH2O_i)**2
    
    return Keq

# No logs
def Ksolver_H2O_v2(mols_i,ions,T_i,cf,kH2O_i,p_mH):
    
    # Convert p_mH to mH
    mH = 10**-p_mH
    
    mols_i[0][ions == 'H' ] = mH
    mols_i[0][ions == 'OH'] = 0
    
    # Get mOH from charge balance
    mOH = aq.balance.charges(mols_i,ions).ravel()
    mols_i[0][ions == 'OH'] = mOH
    
    acfs = pz.model.acfs(mols_i,ions,T_i,cf).ravel()
    
    gH  = acfs[ions == 'H' ]
    gOH = acfs[ions == 'OH']
    
    Keq = (gH * mH * gOH * mOH - kH2O_i)**2
    
    return Keq

# Solve for H and OH concentrations!
Keq  = np.full_like(T,np.nan)

go = time()
for i in range(len(T)):

    print(i)
    
    mols_i = np.expand_dims(mols[i],0)
    T_i    = np.expand_dims(T   [i],0)
    ln_kH2O_i = ln_kH2O[i]
    
    p_mH_i = minimize(lambda p_mH: Ksolver_H2O(mols_i,ions,T_i,cf,
                                               ln_kH2O_i,p_mH),
                      7, method='Nelder-Mead')
    
    mols[i,ions == 'H' ] = 10**-p_mH_i['x']
    mols[i,ions == 'OH'] = 10**-p_mH_i['x']
        
    Keq[i] = Ksolver_H2O(mols_i,ions,T_i,cf,ln_kH2O_i,p_mH_i['x'])
print(time() - go)

mols2 = deepcopy(mols)
Keq2 = np.full_like(T,np.nan)

go = time()
for i in range(len(T)):

    print(i)
    
    mols2_i = np.expand_dims(mols2[i],0)
    T_i     = np.expand_dims(T    [i],0)
    kH2O_i  = kH2O[i]
    
    p_mH_i2 = minimize(lambda p_mH: Ksolver_H2O_v2(mols2_i,ions,T_i,cf,
                                                   kH2O_i,p_mH),
                       7, method='Nelder-Mead')
    
    mols2[i,ions == 'H' ] = 10**-p_mH_i2['x']
    mols2[i,ions == 'OH'] = 10**-p_mH_i2['x']
        
    Keq2[i] = Ksolver_H2O_v2(mols2_i,ions,T_i,cf,kH2O_i,p_mH_i2['x'])
print(time() - go)

# Calculate pH and print result
p_mH  = -np.log10(mols [:,ions == 'H'])
p_mH2 = -np.log10(mols2[:,ions == 'H'])

print(p_mH, p_mH2)

#%% Get arrays of Keq values


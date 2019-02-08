from copy import deepcopy
import aqualibrium as aq
import numpy  as np
import pytzer as pz

from scipy.optimize import minimize

# Get data with equilib
tots,mols,eles,ions,T = aq.io.gettots('data/CRP94 Table 8.csv')
cf = deepcopy(pz.cfdicts.CRP94)

# Calculate HSO4 thermodynamic dissociation constant
ln_kHSO4 = aq.dissoc.HSO4_CRP94(T)[0]
kHSO4 = np.exp(ln_kHSO4)

# Add H-HSO4-SO4 to model
mols = np.concatenate((mols,np.zeros((np.shape(mols)[0],3))), axis=1)
ions = np.concatenate((ions,np.array(['H','HSO4','SO4'])))

# Expand cfdict
cf.add_zeros(ions)

# Define solver function
def Ksolver_H2SO4(tots_i,mols_i,eles,ions,T_i,cf,ln_kHSO4_i,p_mH):
    
    # Convert p_mH to mH
    mH = 10**-p_mH
    
    # Calculate concentrations
    tH2SO4 = tots_i[0][eles == 't_H2SO4']
    mHSO4 = 2*tH2SO4 - mH
    mSO4  = mH - tH2SO4
    
    # Assign concentrations
    mols_i = deepcopy(mols_i)
    mols_i[0][ions == 'H'   ] = mH
    mols_i[0][ions == 'HSO4'] = mHSO4
    mols_i[0][ions == 'SO4' ] = mSO4
    
    ln_acfs = pz.model.ln_acfs(mols_i,ions,T_i,cf).ravel()
    
    ln_gH    = ln_acfs[ions == 'H'   ]
    ln_gHSO4 = ln_acfs[ions == 'HSO4']
    ln_gSO4  = ln_acfs[ions == 'SO4' ]
    
    Keq = (ln_gH + np.log(mH) + ln_gSO4 + np.log(mSO4) \
        - ln_gHSO4 - np.log(mHSO4) - ln_kHSO4_i)**2
    
    return Keq

# Define non-log solver function
def Ksolver_H2SO4_v2(tots_i,mols_i,eles,ions,T_i,cf,kHSO4_i,p_mH):
    
    # Convert p_mH to mH
    mH = 10**-p_mH
    
    # Calculate concentrations
    tH2SO4 = tots_i[0][eles == 't_H2SO4']
    mHSO4 = 2*tH2SO4 - mH
    mSO4  = mH - tH2SO4
    
    # Assign concentrations
    mols_i = deepcopy(mols_i)
    mols_i[0][ions == 'H'   ] = mH
    mols_i[0][ions == 'HSO4'] = mHSO4
    mols_i[0][ions == 'SO4' ] = mSO4
    
    acfs = pz.model.acfs(mols_i,ions,T_i,cf).ravel()
    
    gH    = acfs[ions == 'H'   ]
    gHSO4 = acfs[ions == 'HSO4']
    gSO4  = acfs[ions == 'SO4' ]
    
    Keq = (gH * mH * gSO4 * mSO4 / (gHSO4 * mHSO4) - kHSO4_i)**2
    
    return Keq

# Solve for H and OH concentrations!
Keq = np.full_like(T,np.nan)
solvtype = 'v2'

for i in range(5):#range(len(T)):

    print(i)
    
    tots_i = np.expand_dims(tots[i],0)
    mols_i = np.expand_dims(mols[i],0)
    T_i    = np.expand_dims(T   [i],0)
    ln_kHSO4_i = ln_kHSO4[i]
    kHSO4_i    =    kHSO4[i]
    
    p_mH_g = -np.log10(tots_i[0][eles == 't_H2SO4']*1.5)
    
    
    if solvtype == 'v1':
        p_mH_i = minimize(lambda p_mH: Ksolver_H2SO4(tots_i,mols_i,eles,ions,
                                                     T_i,cf,ln_kHSO4_i,p_mH),
                          p_mH_g, method='Nelder-Mead')
        
        Keq[i] = Ksolver_H2SO4(tots_i,mols_i,eles,ions,T_i,cf,ln_kHSO4_i,
                               p_mH_i['x'])
        
    elif solvtype == 'v2':
        p_mH_i = minimize(lambda p_mH: Ksolver_H2SO4_v2(tots_i,mols_i,eles,
                                                        ions,T_i,cf,kHSO4_i,
                                                        p_mH),
                          p_mH_g, method='Nelder-Mead')
        
        Keq[i] = Ksolver_H2SO4_v2(tots_i,mols_i,eles,ions,T_i,cf,kHSO4_i,
                                  p_mH_i['x'])

    

    mH_i    = 10**-p_mH_i['x']
    mHSO4_i = mH_i - tots_i
    mSO4_i  = tots_i - mHSO4_i

    mols[i,ions == 'H'   ] = mH_i
    mols[i,ions == 'HSO4'] = mHSO4_i
    mols[i,ions == 'SO4' ] = mSO4_i
    
a_HSO4 = mols[:,ions == 'HSO4'] / tots[:,eles == 't_H2SO4']


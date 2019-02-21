from copy import deepcopy
import aqualibrium as aq
import numpy  as np
import pytzer as pz

from scipy.optimize import minimize

# Get data with aqualibrium
#tots,mols,eles,ions,T = aq.io.gettots('data/CRP94 Table 8.csv')
tots,mols,eles,ions,T = aq.io.gettots('data/aquaQuickStartX.csv')
#cf = deepcopy(pz.cfdicts.CRP94)
cf = deepcopy(pz.cfdicts.MarChemSpec)

## Switch temperature
#T[:] = 273.15

# Calculate thermodynamic dissociation constants
ln_kHSO4 = aq.dissoc.HSO4_CRP94(T)[0]
ln_kH2O  = aq.dissoc.H2O_M88   (T)[0]

kHSO4 = np.exp(ln_kHSO4)
kH2O  = np.exp(ln_kH2O)

# Add H-HSO4-SO4-OH to model
mols = np.concatenate((mols,np.zeros((np.shape(mols)[0],4))), axis=1)
ions = np.concatenate((ions,np.array(['H','HSO4','SO4','OH'])))

## Add some OH interaction terms
#cf.theta['HSO4-OH'] = pz.coeffs.theta_HSO4_OH_HMW84
#cf.theta['OH-SO4' ] = pz.coeffs.theta_OH_SO4_HMW84
#
#cf.psi['H-HSO4-OH'] = pz.coeffs.psi_H_HSO4_OH_HMW84
#cf.psi['H-OH-SO4' ] = pz.coeffs.psi_H_OH_SO4_HMW84

# Expand cfdict
cf.add_zeros(ions)

# Define log solver function with OH
def Ksolver_H2SO4_H2O_v1(p_mOH,tots_i,eles,dHSO4,mols_i,ions,T_i,kH2O_i,kHSO4_i):
    
    # Convert p_mOH to mOH
    mOH = 10**-p_mOH
        
    # Calculate concentrations
    tH2SO4 = tots_i[0][eles == 't_H2SO4']
    
    mHSO4 = np.maximum(np.minimum(tH2SO4 * dHSO4,tH2SO4),0)
    mSO4  = tH2SO4 - mHSO4
    
    # Put concentrations into molality array
    Kmols = deepcopy(mols_i)
    Kmols[0][ions == 'OH'  ] = mOH
    Kmols[0][ions == 'HSO4'] = mHSO4
    Kmols[0][ions == 'SO4' ] = mSO4
    
    # Calculate mH from charge balance and put into molality array
    mH = -aq.balance.charges(Kmols,ions)[0]
    Kmols[0][ions == 'H'] = mH
    
    # Calculate activities
    ln_acfs = pz.model.ln_acfs(Kmols,ions,T_i,cf)[0]
    
    # Assign activity coefficients
    ln_gH    = ln_acfs[ions == 'H'   ]
    ln_gHSO4 = ln_acfs[ions == 'HSO4']
    ln_gSO4  = ln_acfs[ions == 'SO4' ]
    ln_gOH   = ln_acfs[ions == 'OH'  ]
    
    # Define K equations
    KH2O = ln_gH + np.log(mH) + ln_gOH + np.log(mOH) - ln_kH2O_i
    with np.errstate(divide='ignore'):
        KH2SO4 = ln_gH + np.log(mH) + ln_gSO4 + np.log(mSO4) \
            - ln_gHSO4 - np.log(mHSO4) - ln_kHSO4_i
        
    Keq = KH2O**2 + KH2SO4**2
    
    return Keq, Kmols, KH2O, KH2SO4


# Define log solver function with H or OH
def Ksolver_H2SO4_H2O(p_mX,tots_i,eles,dHSO4,mols_i,ions,T_i,
                      ln_kH2O_i,ln_kHSO4_i):

    Kmols = deepcopy(mols_i)
    
    # Calculate concentrations
    tH2SO4 = tots_i[0][eles == 't_H2SO4']
    
#    mHSO4 = np.maximum(np.minimum(tH2SO4 * dHSO4,tH2SO4),0)
    mHSO4 = tH2SO4 * (np.tanh(dHSO4) + 1) / 2
    mSO4  = tH2SO4 - mHSO4
    
    # Put other concentrations into molality array
    Kmols[0][ions == 'HSO4'] = mHSO4
    Kmols[0][ions == 'SO4' ] = mSO4
    
    # Determine if X in p_mX should be H or OH from charge balance
    #  (helps to avoid solutions with negative concentrations)
    zeq = aq.balance.charges(Kmols,ions)[0]
    
    if zeq < 0: # acidic solution

        # Convert p_mX to mOH
        mOH = 10**-p_mX
        Kmols[0][ions == 'OH'] = mOH
    
        # Calculate mH from charge balance and put into molality array
        mH = -aq.balance.charges(Kmols,ions)[0]
        Kmols[0][ions == 'H'] = mH
        
    elif zeq >= 0: # basic or neutral solution
        
        # Convert p_mX to mH
        mH = 10**-p_mX
        Kmols[0][ions == 'H'] = mH
    
        # Calculate mOH from charge balance and put into molality array
        mOH = aq.balance.charges(Kmols,ions)[0]
        Kmols[0][ions == 'OH'] = mOH
    
    # Calculate activities
    ln_acfs = pz.model.ln_acfs(Kmols,ions,T_i,cf)[0]
    
    # Assign activity coefficients
    ln_gH    = ln_acfs[ions == 'H'   ]
    ln_gHSO4 = ln_acfs[ions == 'HSO4']
    ln_gSO4  = ln_acfs[ions == 'SO4' ]
    ln_gOH   = ln_acfs[ions == 'OH'  ]
    
    # Define K equations
    KH2O = ln_gH + np.log(mH) + ln_gOH + np.log(mOH) - ln_kH2O_i
    if tH2SO4 > 0:
        with np.errstate(divide='ignore'):
            KH2SO4 = ln_gH + np.log(mH) + ln_gSO4 + np.log(mSO4) \
                - ln_gHSO4 - np.log(mHSO4) - ln_kHSO4_i
    else:
        KH2SO4 = 0
        
    Keq = KH2O**2 + KH2SO4**2
    
    return Keq, Kmols, KH2O, KH2SO4


Ksolved = np.full((np.size(T),2),np.nan)

for i in range(len(T)):
    
    print(i)

    tots_i = np.expand_dims(tots[i],0)
    mols_i = np.expand_dims(mols[i],0)
    T_i    = np.expand_dims(T   [i],0)
    
    ln_kHSO4_i = ln_kHSO4[i]
    ln_kH2O_i  = ln_kH2O [i]
    
    dHSO4 = 0
    
    p_mOH = 7.0
    
    Ksolved_i = minimize(lambda Ksolved: \
        Ksolver_H2SO4_H2O(Ksolved[0],tots_i,eles,Ksolved[1],
                          mols_i,ions,T_i,ln_kH2O_i,ln_kHSO4_i)[0],
        [p_mOH,dHSO4], method='Nelder-Mead',
        options={'disp': True, 'adaptive': True})
        
    Ksolved[i] = Ksolved_i['x']
    
    mols[i] = Ksolver_H2SO4_H2O(Ksolved[i][0],tots_i,eles,Ksolved[i][1],
                                mols_i,ions,T_i,ln_kH2O_i,ln_kHSO4_i)[1][0]
    
    print(Ksolved_i['x'])
    
#%% Get stoichiometric equilibrium constant
Kst = mols[:,ions=='H'] * mols[:,ions=='SO4'] / mols[:,ions=='HSO4']

acfs = pz.model.acfs(mols,ions,T,cf)
    
Kst_acfs = kHSO4 * acfs[:,ions=='HSO4'] \
    / (acfs[:,ions=='H'] * acfs[:,ions=='SO4'])

# Print out pmH
print(-np.log10(mols[:,ions == 'H']))

#%% Save for MATLAB
#from scipy.io import savemat
#savemat('mat/H2SO4_H2O_CRP94.mat',
#        {'tots': tots,
#         'eles': eles,
#         'Ksolved': Ksolved,
#         'mols': mols,
#         'ions': ions})

#%%
#G_p_mOH = np.linspace(12,14,21)
#G_dHSO4 = np.linspace(0.6,0.999,21)
#
#G_p_mOH,G_dHSO4 = np.meshgrid(G_p_mOH,G_dHSO4)
#
#G_KH2O   = np.full_like(G_p_mOH,np.nan)
#G_KH2SO4 = np.full_like(G_p_mOH,np.nan)
#G_Keq    = np.full_like(G_p_mOH,np.nan)
#
#for i in range(np.size(G_Keq)):
#    
#    if i % 50 == 0:
#        print(i)
#            
#    G_Keq.ravel()[i],_,G_KH2O.ravel()[i],G_KH2SO4.ravel()[i] \
#                = Ksolver_H2SO4_H2O(G_p_mOH.ravel()[i],tots_i,eles,
#                                    G_dHSO4.ravel()[i],
#                                    mols_i,ions,T_i,kH2O_i,kHSO4_i)
#                
#
##%%
#from scipy.io import savemat
#
#savemat('mat/H2SO4_H2O.mat',
#        {'p_mOH' : G_p_mOH ,
#         'dHSO4' : G_dHSO4 ,
#         'Keq'   : G_Keq   ,
#         'KH2O'  : G_KH2O  ,
#         'KH2SO4': G_KH2SO4,
#         'solv'  : Ksolved_i['x']})

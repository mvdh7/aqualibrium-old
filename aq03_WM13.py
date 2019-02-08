import aqualibrium as aq
import pytzer as pz
from copy import deepcopy
import numpy as np

from scipy.optimize import minimize

test = aq.dissoc.MgOH_CW91(298.15)

# Get data with aqualibrium
tots,mols,eles,ions,T = aq.io.gettots('data/aquaQuickStart.csv')
cf = deepcopy(pz.cfdicts.MarChemSpec)

# Calculate thermodynamic dissociation constants
ln_kHSO4  = aq.dissoc.HSO4_CRP94(T)[0]
ln_kH2O   = aq.dissoc.H2O_M88   (T)[0]
ln_ktrisH = aq.dissoc.trisH_BH61(T)[0]
ln_kMgOH  = aq.dissoc.MgOH_CW91 (T)[0]

# Add extra ions to model
eleions = np.array(['H','HSO4','SO4','OH','trisH','MgOH','Mg'])
mols = np.concatenate((mols,np.zeros((np.shape(mols)[0],np.size(eleions)))), 
                      axis=1)
ions = np.concatenate((ions,eleions))

# Flesh out cfdict, just in case
cf.add_zeros(ions)

# Define solver function
def Gsolver(p_mX,tots_i,eles,dHSO4,mols_i,ions,T_i,ln_kH2O_i,ln_kHSO4_i,
            ln_kMgOH_i,dMgOH):
    
    Kmols = deepcopy(mols_i)
    
    # Calculate sulfate concentrations
    tH2SO4 = tots_i[0][eles == 't_H2SO4']
    
    mHSO4 = np.maximum(np.minimum(tH2SO4 * dHSO4,tH2SO4),0)
    mSO4  = tH2SO4 - mHSO4
    
    # Calculate tris concentrations
    ttris = tots_i[0][eles == 't_tris'] * 0
    
    dtrisH = 0
    
    mtrisH = np.maximum(np.minimum(ttris * dtrisH,ttris),0)
    mtris  = ttris - mtrisH
    
    # Calculate Mg concentrations
    tMg = tots_i[0][eles == 't_Mg']
    
    mMgOH = np.maximum(np.minimum(tMg * dMgOH,tMg),0)
    mMg   = tMg - mMgOH
    
    # Put other concentrations into molality array
    Kmols[0][ions == 'HSO4' ] = mHSO4
    Kmols[0][ions == 'SO4'  ] = mSO4
    
    Kmols[0][ions == 'trisH'] = mtrisH
    Kmols[0][ions == 'tris' ] = mtris
    
    Kmols[0][ions == 'MgOH' ] = mMgOH
    Kmols[0][ions == 'Mg'   ] = mMg
    
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
    ln_gMg   = ln_acfs[ions == 'Mg'  ]
    ln_gMgOH = ln_acfs[ions == 'MgOH']
    
    # Define K equations
    KH2O = ln_gH + np.log(mH) + ln_gOH + np.log(mOH) - ln_kH2O_i
    
    if tH2SO4 > 0:
        with np.errstate(divide='ignore'):
            KH2SO4 = ln_gH + np.log(mH) + ln_gSO4 + np.log(mSO4) \
                - ln_gHSO4 - np.log(mHSO4) - ln_kHSO4_i
    else:
        KH2SO4 = 0
        
    if tMg > 0:
        with np.errstate(divide='ignore'):
            KMgOH = ln_gOH + np.log(mOH) + ln_gMg + np.log(mMg) \
                - ln_gMgOH - np.log(mMgOH) - ln_kMgOH_i
    else:
        KMgOH = 0
        
    Keq = KH2O**2 + KH2SO4**2 + KMgOH**2
    
    return Keq, Kmols, KH2O, KH2SO4

Gsolved = np.full((np.size(T),3),np.nan)

for i in range(len(T)):
    
    print(i)

    tots_i = np.expand_dims(tots[i],0)
    mols_i = np.expand_dims(mols[i],0)
    T_i    = np.expand_dims(T   [i],0)
    
    ln_kHSO4_i = ln_kHSO4[i]
    ln_kH2O_i  = ln_kH2O [i]
    ln_kMgOH_i = ln_kMgOH[i]
    
    dHSO4 = 0.5
    dMgOH = 0.5
    p_mX  = 7.0
    
#    Gguess = Gsolver(p_mX,tots_i,eles,dHSO4,
#                     mols_i,ions,T_i,ln_kH2O_i,ln_kHSO4_i,
#                     ln_kMgOH_i,dMgOH)[1][0]
    
    Gsolved_i = minimize(lambda Gsolved: \
        Gsolver(Gsolved[0],tots_i,eles,Gsolved[1],
                mols_i,ions,T_i,ln_kH2O_i,ln_kHSO4_i,ln_kMgOH_i,
                Gsolved[2])[0],
        [p_mX,dHSO4,dMgOH], method='Nelder-Mead',
        options={'disp': True, 'adaptive': True})
        
    Gsolved[i] = Gsolved_i['x']
    
    mols[i] = Gsolver(Gsolved[i][0],tots_i,eles,Gsolved[i][1],
                      mols_i,ions,T_i,ln_kH2O_i,ln_kHSO4_i,
                      ln_kMgOH_i,Gsolved[i][2])[1][0]
    
    print(Gsolved_i['x'])

print(-np.log10(mols[:,ions == 'H']))

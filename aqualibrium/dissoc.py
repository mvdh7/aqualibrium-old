# aqualibrium: Equilibrium solver for aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from numpy import exp, float_, log, log10, logical_and
from pytzer.coeffs import M88_eq13

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Moller (1988) ~~~~~
         
def H2O_M88(T):
    
    ln_kH2O = M88_eq13(T,float_([ 
         1.04031130e+3,
         4.86092851e-1,
        -3.26224352e+4,
        -1.90877133e+2,
        -5.35204850e-1,
        -2.32009393e-4,
         5.20549183e+1,
         0            ]))
    
    valid = logical_and(T >= 298.15, T <= 523.15)
    
    return ln_kH2O, valid


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Clegg et al. (1994) ~~~~~

def HSO4_CRP94(T):
    
    # CRP94 Eq. (21) with extra digits on constant term
    #  (S.L. Clegg, pers. comm, 7 Feb 2019)
    log10_kHSO4 =   562.694864456          \
                -   102.5154      * log(T) \
                -     1.117033e-4 * T**2   \
                +     0.2477538   * T      \
                - 13273.75        / T
    
    valid = logical_and(T >= 273.15, T <= 328.15)
    
    # Convert log base
    ln_kHSO4 = log10_kHSO4 / log10(exp(1))
    
    return ln_kHSO4, valid


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Clegg & Whitfield (1991) ~~~~~
    
def MgOH_CW91(T):
    
    # CW91 Eq. (244) [p392]
    ln_kMgOH = 8.9108 - 1155/T
    
    valid = logical_and(T >= 278.15, T <= 308.15)
    
    return ln_kMgOH, valid


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Bates & Hetzer (1961) ~~~~~
    
def trisH_BH61(T):
    
    # BH61 Eq. (3)
    ln_ktrisH = 2981.4 / T - 3.5888 + 0.005571 * T
    
    valid = logical_and(T >= 273.15, T <= 323.15)
    
    return ln_ktrisH, valid

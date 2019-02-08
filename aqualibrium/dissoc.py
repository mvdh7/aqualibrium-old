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
    
    # CRP94 Eq. (21)
    log10_kHSO4 =   562.69486              \
                -   102.5154      * log(T) \
                -     1.117033e-4 * T**2   \
                +     0.2477538   * T      \
                - 13273.75        / T
    
    valid = logical_and(T >= 273.15, T <= 328.15)
    
    return log10_kHSO4 / log10(exp(1)), valid

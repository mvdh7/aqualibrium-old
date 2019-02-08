# aqualibrium: Equilibrium solver for aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from numpy import expand_dims, vstack
from numpy import sum as np_sum
from pytzer.props import charges as pz_props_charges

def charges(mols,ions):

    # Evaluate charges
    zs = expand_dims(pz_props_charges(ions)[0],0)
    
    # Calculate charge balance
    ZB = vstack(np_sum(mols * zs, axis=1))
    
    return ZB

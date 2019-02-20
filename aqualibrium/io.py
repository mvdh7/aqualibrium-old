# aqualibrium: Equilibrium solver for aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from numpy import array, genfromtxt, nan_to_num, shape

#==============================================================================
#============================================== Import concentration data =====

def gettots(filename, delimiter=','):

    data = genfromtxt(filename, delimiter=delimiter, skip_header=1)
    head = genfromtxt(filename, delimiter=delimiter, dtype='U',
                      skip_footer=shape(data)[0])

    nan_to_num(data, copy=False)

    # Get temperatures
    TL = head == 'tempK'
    T = data[:,TL]
    
    data = data[:,~TL].transpose()
    head = head[  ~TL]
       
    eles = array([ele for ele in head if 't_'     in ele])
    ions = array([ion for ion in head if 't_' not in ion])
    
    tots = array([tot for i, tot in enumerate(data)    \
                  if 't_'     in head[i]]).transpose()
    
    mols = array([mol for i, mol in enumerate(data)    \
                  if 't_' not in head[i]]).transpose()

    return tots, mols, eles, ions, T

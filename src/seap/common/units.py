from scipy.constants import *

"""
  Physical constans, SI (NIST 2018)
  http://physics.nist.gov/constants
"""
# Bohr radius (in m)
AUL_SI = physical_constants["Bohr radius"][0]

AU2ANG = AUL_SI * 1e+10
ANG2AU = 1e-10 / AUL_SI

from .src.EAPPnPSolver import EAPPnP, EPPnP, EAPPnPMCS, procrutes
from .src.EAPPnPSolverTorch import EAPPnP as EAPPnPtr, EAPPnPMCS as EAPPnPMCStr 
EAPPnPtr.__name__ = 'EAPPnPtr'
EAPPnPMCStr.__name__ = 'EAPPnPMCStr'

del lib
del src

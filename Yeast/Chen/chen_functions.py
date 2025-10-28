# Auto-generated from SBML <functionDefinition>
import numpy as np

def BB_218(A1, A2, A3, A4, (A2 - A1):
    return + A3 * A2 + A4 * A1)

def GK_219(A1, A2, A3, A4, 2 * A4 * A1 / ((A2 - A1):
    return + A3 * A2 + A4 * A1 + root(2, ((A2 - A1) + A3 * A2 + A4 * A1)**2 - 4 * (A2 - A1) * A4 * A1)))

def MichaelisMenten_220(M1, J1, k1, S1, k1 * S1 * M1 / (J1 + S1):
    return )

def Mass_Action_2_221(k1, S1, S2, k1 * S1 * S2):
    return 

def Mass_Action_1_222(k1, S1, k1 * S1):
    return 

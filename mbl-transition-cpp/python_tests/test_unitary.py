import numpy as np
import scipy.linalg as LA
import pytest
import pennylane as qml
from kicked_ising_mbl.kicked_ising_mbl import create_unitary
import math

def circuit(phis, theta):
    for idx in range(len(phis)):
        qml.RX(theta/2, wires=idx)
        
    for idx, phi in enumerate(phis):
        qml.RZ(phi, wires=len(phis)-1-idx)
        
    for i in range(len(phis) - 1):
        qml.CZ(wires=[i,i+1])
    
    for idx in range(len(phis)):
        qml.RX(theta/2, wires=idx)
        
    return qml.state()

@pytest.mark.parametrize("phis, theta", [
    ([0.172, -0.2, 0.3, -0.4], 0.2),
    ([-0.1, 0.283, -0.3, 0.4, -0.6, 0.7], 0.3)
])
def test_unitary(phis, theta):
    m1 = qml.matrix(circuit, wire_order=range(len(phis)))(phis, theta)
    m2 = create_unitary(phis, theta)
    assert(np.allclose(m1, m2))


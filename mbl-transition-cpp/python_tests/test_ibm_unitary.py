import numpy as np
import scipy.linalg as LA
import pytest
import pennylane as qml
from kicked_ising_mbl.kicked_ising_mbl import ibm_unitary
import math

def circuit(phis, theta):
    N = len(phis)

    for i in range(0, len(phis), 2):
        qml.CZ(wires=[N-1-i, N-1-(i+1)])

    for idx in range(len(phis)):
        qml.RX(theta, wires=idx)
        
    for idx, phi in enumerate(phis):
        qml.RZ(phi, wires=len(phis)-1-idx)
    
    for i in range(1, len(phis)-2, 2):
        qml.CZ(wires=[N-1-i, N-1-(i+1)])

    for idx in range(len(phis)):
        qml.RX(theta, wires=idx)

    for idx, phi in enumerate(phis):
        qml.RZ(phi, wires=len(phis)-1-idx)
        
    return qml.state()

@pytest.mark.parametrize("phis, theta", [
    ([0.0, 0.0, 0.0, 0.0], 0.0),
    ([0.0, 0.0, 0.0, 0.0], 0.2),
    ([0.1, -0.2, 0.3, -0.4], 0.0),
    ([0.1, -0.2, 0.3, -0.4], 0.2),
    ([-0.1, 0.2, -0.3, 0.4, -0.6, 0.7], 0.3)
])
def test_unitary(phis, theta):
    m1 = qml.matrix(circuit, wire_order=range(len(phis)))(phis, theta)
    m2 = ibm_unitary(phis, theta)
    print(np.diag(m1))
    print(np.diag(m2))
    assert(np.allclose(m1, m2))


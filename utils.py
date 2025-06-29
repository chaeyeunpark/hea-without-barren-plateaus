import numpy as np

def create_zero_st(num_qubits) :
    assert num_qubits >= 0
    st_0 = np.array([1.0,0.0])
    st_N = st_0
    for i in range(num_qubits - 1) :
        st_N = np.kron(st_N,st_0)

    return st_N

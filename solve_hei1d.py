import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import math
import sys
from numpy.random import Generator, PCG64
import argparse
import json
from utils import create_zero_st
import secrets
from pennylane.operation import Tensor
import scipy.sparse.linalg as LA

LAYERS_PER_BLOCK = 2


def make_ham(num_qubits):
    obs = []
    coeffs = []
    for w in range(num_qubits-1):
        obs.append(qml.PauliX(w) @ qml.PauliX(w+1))
        coeffs.append(1.0)

    for w in range(num_qubits-1):
        obs.append(qml.PauliY(w) @ qml.PauliY(w+1))
        coeffs.append(1.0)

    for w in range(num_qubits-1):
        obs.append(qml.PauliZ(w) @ qml.PauliZ(w+1))
        coeffs.append(1.0)

    for w in range(num_qubits):
        obs.append(qml.PauliZ(w))
        coeffs.append(1.0)

    return qml.Hamiltonian(coeffs, obs)

def create_y_plus(num_qubits) :
    assert num_qubits >= 0
    st_0 = np.array([1.0/math.sqrt(2),1j/math.sqrt(2)])
    st_N = st_0
    for i in range(num_qubits - 1) :
        st_N = np.kron(st_N,st_0)

    return st_N

def create_circuit(num_qubits, num_blocks, obs):
    ini_st = create_y_plus(num_qubits)

    def circuit(x):
        qml.StatePrep(ini_st, wires=range(num_qubits))

        for k in range(num_blocks):
            for w in range(num_qubits):
                qml.RX(x[k, w, 0], wires=w)
                qml.RZ(x[k, w, 1], wires=w)

            for w in range(num_qubits-1):
                qml.CZ(wires=[w,w+1])

        return qml.expval(obs)
    
    return circuit

def constrained_params(rng, *,constant, num_qubits, num_blocks):
    num_params = num_blocks * LAYERS_PER_BLOCK * num_qubits
    init_params = rng.uniform(size= (num_params,))
    init_params = init_params*(2*constant*math.pi)/num_blocks/num_qubits
    return init_params.reshape(num_blocks, num_qubits, 2)

def mbl_params(rng, *, num_qubits, num_blocks):
    rx_params = np.tile(0.10*rng.uniform(size=(num_blocks,1)), num_qubits)
    rz_params = rng.uniform(size = (num_blocks,num_qubits), low=-math.pi, high=math.pi)
    return np.dstack((rx_params, rz_params))

def only_one_is_true(*l):
    return l.count(True) == 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient scaling of hamiltonian variational Ansatz for the XXZ model')
    parser.add_argument('--device', required=True, type=str, choices=["lightning.qubit", "lightning.gpu"], help='Device')
    parser.add_argument('--num_qubits', required=True, type=int, help='Number of qubits (N)')
    parser.add_argument('--num_blocks', required=True, type=int, help='Number of blocks (p)')
    parser.add_argument('--num_iter', required=True, type=int, help='Number of total iteration')
    parser.add_argument('--eta', required=True, type=float, help='Learning rate')
    parser.add_argument('--constant', type=float, help='constant')
    parser.add_argument('--mbl', action='store_true', help='Use MBL initialization')
    parser.add_argument('--random', action='store_true', help="Use random initialization")
    parser.add_argument('--gaussian', action='store_true', help="Use Gaussian initialization")

    args = parser.parse_args()

    if not only_one_is_true(bool(args.random), bool(args.constant), bool(args.gaussian), bool(args.mbl)):
        raise ValueError("Only one of the arguments --constant, --random, and --small must be given")

    device = args.device
    num_blocks = args.num_blocks
    num_qubits = args.num_qubits
    num_iter = args.num_iter
    eta = args.eta

    constant = args.constant

    args_in = vars(args)
    with open('args_in.json', 'w') as f:
        json.dump(args_in, f, indent=4)

    rng = Generator(PCG64(secrets.randbits(128)))

    # Load molecular Hamiltonian from quantum dataset

    num_params = num_blocks * LAYERS_PER_BLOCK * num_qubits

    dev = qml.device(device, wires=num_qubits)
    obs = make_ham(num_qubits)

    h_mat = obs.sparse_matrix()
    """
    print(LA.eigsh(h_mat, which='SA', return_eigenvectors=False))
    """
    circuit = qml.QNode(create_circuit(num_qubits, num_blocks, obs), dev, diff_method="adjoint")

    if args.constant:
        init_params = constrained_params(rng, constant=constant, num_qubits=num_qubits, num_blocks=num_blocks)
    elif args.random:
        init_params = 2*math.pi*rng.random(size = num_params).reshape(num_blocks, num_qubits, 2)
    elif args.gaussian:
        L = 2*num_blocks # for Gaussian initializaion
        gamma = np.sqrt(1/(S*L)) # for Gaussian initialization
        init_params = rng.normal(0, gamma, size = num_params)
    elif args.mbl:
        init_params = mbl_params(rng, num_qubits=num_qubits, num_blocks=num_blocks)
    else:
        raise ValueError("Error!")

    params = pnp.array(init_params, requires_grad=True)
    opt = qml.AdamOptimizer(eta, beta1=0.9, beta2=0.999, eps=1e-7)

    for step in range(num_iter):
        cost = circuit(params)
        params = opt.step(circuit, params)
        print(f"step: {step}, cost: {cost}", flush=True)

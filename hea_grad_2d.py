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

LAYERS_PER_BLOCK = 2

def create_circuit(Lx, Ly, num_blocks, obs):
    num_qubits = Lx*Ly
    def circuit(x):
        for k in range(num_blocks):
            for w in range(num_qubits):
                qml.RX(x[LAYERS_PER_BLOCK*(num_qubits*k + w)+0], wires=w)
                qml.RZ(x[LAYERS_PER_BLOCK*(num_qubits*k + w)+1], wires=w)

            for i in range(Lx):
                for j in range(Ly-1):
                    idx0 = j*Lx + i
                    idx1 = (j+1)*Lx + i
                    qml.CZ(wires=[idx0, idx1])

            for i in range(Lx-1):
                for j in range(Ly):
                    idx0 = j*Lx + i
                    idx1 = j*Lx + i + 1
                    qml.CZ(wires=[idx0, idx1])

        return qml.expval(obs)
    
    return circuit

def constrained_params(rng, *, constant, num_qubits, num_blocks):
    num_params = num_blocks * LAYERS_PER_BLOCK * num_qubits
    init_params = rng.uniform(size = num_params)
    init_params *= (2*constant*math.pi)/num_blocks/num_qubits

    return init_params

def only_one_is_true(*l):
    return l.count(True) == 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient scaling of hamiltonian variational Ansatz for the XXZ model')
    parser.add_argument('--device', required=True, type=str, choices=["lightning.qubit", "lightning.gpu"], help='Device')
    parser.add_argument('--lattice', required=True, type=str, help='The size of the square lattice')
    parser.add_argument('--obs', required=True, type=str, choices=['single', 'multi'], help='The type of the observable')
    parser.add_argument('--num_blocks', required=True, type=int, help='Number of blocks (p)')
    parser.add_argument('--num_iter', required=True, type=int, help='Number of total iteration')
    parser.add_argument('--constant', type=float, help='constant')
    parser.add_argument('--random', action='store_true', help="Use random initialization")
    parser.add_argument('--gaussian', action='store_true', help="Use Gaussian initialization")

    args = parser.parse_args()

    if not only_one_is_true(bool(args.random), bool(args.constant), bool(args.gaussian)):
        raise ValueError("Only one of the arguments --constant, --random, and --small must be given")

    device = args.device
    num_blocks = args.num_blocks
    num_iter = args.num_iter

    constant = args.constant

    lattice = args.lattice.split('x')
    Lx = int(lattice[0])
    Ly = int(lattice[1])

    num_qubits = Lx * Ly

    if args.obs == 'single':
        obs = qml.PauliY(0)
        S = 1
    else:
        ops = [qml.PauliY(0)]
        for i in range(1, num_qubits):
            ops.append(qml.PauliZ(i))

        obs = Tensor(*ops)
        S = num_qubits # for Gaussian initialization

    print("obs={}".format(str(obs)))

    args_in = vars(args)
    with open('args_in.json', 'w') as f:
        json.dump(args_in, f, indent=4)

    rng = Generator(PCG64(secrets.randbits(128)))

    dev = qml.device(device, wires=num_qubits)
    circuit = qml.QNode(create_circuit(Lx, Ly, num_blocks, obs), dev, diff_method="adjoint")

    num_params = num_blocks * LAYERS_PER_BLOCK * num_qubits
    grads = np.zeros((num_iter, num_params), dtype=np.float128)

    L = 2*num_blocks # for Gaussian initializaion
    gamma = np.sqrt(1/(S*L)) # for Gaussian initialization
                    
    for i in range(num_iter):
        if args.constant:
            param = constrained_params(rng, constant=constant, num_qubits=num_qubits, num_blocks=num_blocks)
        elif args.random:
            param = 2*math.pi*rng.random(size = num_params)
        elif args.gaussian:
            param = rng.normal(0, gamma, size = num_params)
        else:
            raise ValueError("Error!")
        param = pnp.array(param, requires_grad=True)

        grad = qml.grad(circuit)(param)
        grads[i,:] = grad

    np.save("grads.npy", grads)

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

def create_circuit(num_qubits, num_blocks, obs):
    def circuit(x):
        for k in range(num_blocks):
            for w in range(num_qubits):
                qml.RX(x[LAYERS_PER_BLOCK*(num_qubits*k + w)+0], wires=w)
                qml.RZ(x[LAYERS_PER_BLOCK*(num_qubits*k + w)+1], wires=w)
            for w in range(num_qubits-1):
                qml.CZ(wires=[w,w+1])

        return qml.expval(obs)
    
    return circuit

def constrained_params(rng, *, constant, num_qubits, num_blocks):
    num_params = num_blocks * LAYERS_PER_BLOCK * num_qubits
    init_params = rng.uniform(size = num_params)
    init_params *= (2*constant*math.pi)/num_blocks/num_qubits

    return init_params

def mbl_params(rng, *, num_qubits, num_blocks):
    init_params = np.zeros((num_blocks, num_qubits, 2), dtype=np.float64)
    init_params[:,:,0] = 0.10*rng.uniform(size=num_blocks).reshape(-1,1)
    init_params[:,:,1] = rng.uniform(size = (num_blocks, num_qubits), low=-math.pi, high=math.pi)

    return init_params.flatten()


def only_one_is_true(*l):
    return l.count(True) == 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient scaling of hamiltonian variational Ansatz for the XXZ model')
    parser.add_argument('--device', required=True, type=str, choices=["lightning.qubit", "lightning.gpu"], help='Device')
    parser.add_argument('--num_qubits', required=True, type=int, help='Number of qubits (N)')
    parser.add_argument('--num_blocks', required=True, type=int, help='Number of blocks (p)')
    parser.add_argument('--obs', required=True, type=str, choices=['single', 'multi'], help='The type of the observable')
    parser.add_argument('--num_iter', required=True, type=int, help='Number of total iteration')
    parser.add_argument('--constant', type=float, help='constant')
    parser.add_argument('--mbl', action='store_true', help='Use MBL initialization')
    parser.add_argument('--random', action='store_true', help="Use random initialization")
    parser.add_argument('--gaussian', action='store_true', help="Use Gaussian initialization")

    args = parser.parse_args()

    if not only_one_is_true(bool(args.random), bool(args.constant), bool(args.gaussian), bool(args.mbl)):
        raise ValueError("Only one of the arguments --constant, --random, and --small must be given")

    device = args.device
    num_qubits = args.num_qubits
    num_blocks = args.num_blocks
    num_iter = args.num_iter

    constant = args.constant

    num_params = num_blocks * LAYERS_PER_BLOCK * num_qubits

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
    circuit = qml.QNode(create_circuit(num_qubits, num_blocks, obs), dev, diff_method="adjoint")

    grads = np.zeros((num_iter,num_params), dtype=np.float128)

    for i in range(num_iter):
        if args.constant:
            param = constrained_params(rng, constant=constant, num_qubits=num_qubits, num_blocks=num_blocks)
        elif args.random:
            param = 2*math.pi*rng.random(size = num_params)
        elif args.gaussian:
            L = 2*num_blocks
            gamma = np.sqrt(1/(S*L))
            param = rng.normal(0, gamma, size = num_params)
        elif args.mbl:
            param = mbl_params(rng, num_qubits=num_qubits, num_blocks=num_blocks)
        else:
            raise ValueError("Error!")
        param = pnp.array(param, requires_grad=True)

        grad = qml.grad(circuit)(param)
        grads[i,:] = grad

    np.save("grads.npy", grads)

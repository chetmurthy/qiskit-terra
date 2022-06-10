import time
import numpy as np
import scipy.sparse as scisparse
from qiskit.quantum_info.operators.symplectic.pauli import *

def timer(msg, f):
    t0 = time.time()
    print("START %s" % (msg,))
    rv = f()
    t1 = time.time()
    print('END ELAPSED %s: %.03fms' % (msg, 1000 * (t1-t0)))
    return rv

def sparse_to_matrix(z,x,phase=0,group_phase=False):
    from qiskit._accelerate.base_pauli import make_data, old_make_data, timed_make_data, timed_old_make_data
    p2 = make_data(z,x, phase, group_phase)
    from scipy.sparse import csr_matrix
    num_qubits = z.size
    dim = 2**num_qubits
    rv = csr_matrix(p2, shape=(dim, dim), dtype=complex)
    return rv

ops = [
    Pauli('IIIIIIIIIIIIIIIIIIYXXY'),
    Pauli('IIIIIIIIIIIIIIIIIIYYYY'),
    Pauli('IIIIIIIIIIIIIIIIIIXXYY'),
    Pauli('IIIIIIIIIIIIIIIIIIYYXX'),
    Pauli('IIIIIIIIIIIIIIIIIIXXXX'),
    Pauli('IIIIIIIIIIIIIIIIIIXYYX'),
    Pauli('IIIIIIIIIIIIIIIIIYZXXY'),
    Pauli('IIIIIIIIIIIIIIIIIYZYYY'),
    Pauli('IIIIIIIIIIIIIIIIIXZXYY'),
    Pauli('IIIIIIIIIIIIIIIIIYZYXX'),
]

op = ops[0]

timer("to_matrix", lambda: op.to_matrix(sparse=True))

timer("_to_matrix", lambda: BasePauli._to_matrix(op.z, op.x, op._phase[0], sparse=True))

timer("_to_matrix0", lambda: BasePauli._to_matrix0(op.z, op.x, op._phase[0], sparse=True))

mat = timer("sparse_to_matrix", lambda: sparse_to_matrix(op.z, op.x, op._phase[0]))
print(repr(mat))

import timeit

print ("args: %s" % ((op.z, op.x, op._phase[0]),))
print(timeit.timeit(lambda: sparse_to_matrix(op.z, op.x, op._phase[0]), number=10))

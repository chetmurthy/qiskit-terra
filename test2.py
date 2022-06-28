import time
import numpy as np
import scipy.sparse as scisparse
from qiskit.quantum_info.operators.symplectic import *
from qiskit.quantum_info.operators.symplectic.base_pauli import *
from qiskit.quantum_info.operators.symplectic.pauli import *

def timer(msg, f):
    t0 = time.time()
    print("START %s" % (msg,))
    rv = f()
    t1 = time.time()
    print('END ELAPSED %s: %.03fms' % (msg, 1000 * (t1-t0)))
    return rv

def sparse_to_matrix(z,x,coeff=1.0 + 0j, phase=0,group_phase=False):
    from qiskit._accelerate.base_pauli import make_data, old_make_data, timed_make_data, timed_old_make_data
    p2 = make_data(z,x, coeff, phase, group_phase)
    from scipy.sparse import csr_matrix
    num_qubits = z.size
    dim = 2**num_qubits
    rv = csr_matrix(p2, shape=(dim, dim), dtype=complex)
    return rv

def qrusty_to_matrix(label, coeff=1.0+0.0j):
    from qiskit._accelerate.base_pauli import qrusty_Pauli_make_data
    (num_qubits, data, indices, indptr) = qrusty_Pauli_make_data(label, coeff)
    print("sparse_to_matrix:\n\tnum_qubits: %s\n\tdata: %s\n\tindices: %s\n\tindptr: %s" %
          (num_qubits, data, indices, indptr))
    from scipy.sparse import csr_matrix
    dim = 2**num_qubits
    rv = csr_matrix((data, indices, indptr), shape=(dim, dim), dtype=complex)
    return rv

def equality_test(p1, p2):
    (data1, indices1, indptr1) = p1
    (data2, indices2, indptr2) = p2
    return np.array_equal(data1,data2) and np.array_equal(indices1,indices2) and np.array_equal(indptr1,indptr2) 

def spmat_equal(m1, m2):
    if m1.shape != m2.shape: return False
    if not np.array_equal(m1.data,m2.data): return False
    if not np.array_equal(m1.indices,m2.indices): return False
    if not np.array_equal(m1.indptr,m2.indptr): return False
    return True

oplist = PauliList([
    'IIIIIIIIIIIIIIIIIIYXXY', 'IIIIIIIIIIIIIIIIIIYYYY',
    'IIIIIIIIIIIIIIIIIIXXYY', 'IIIIIIIIIIIIIIIIIIYYXX',
    'IIIIIIIIIIIIIIIIIIXXXX', 'IIIIIIIIIIIIIIIIIIXYYX',
    'IIIIIIIIIIIIIIIIIYZXXY', 'IIIIIIIIIIIIIIIIIYZYYY',
    'IIIIIIIIIIIIIIIIIXZXYY', 'IIIIIIIIIIIIIIIIIYZYXX',
    'IIIIIIIIIIIIIIIIIXZXXX', 'IIIIIIIIIIIIIIIIIXZYYX',
    'IIIIIIIIIIIIIIIIYZZXXY', 'IIIIIIIIIIIIIIIIYZZYYY',
    'IIIIIIIIIIIIIIIIXZZXYY', 'IIIIIIIIIIIIIIIIYZZYXX',
    'IIIIIIIIIIIIIIIIXZZXXX', 'IIIIIIIIIIIIIIIIXZZYYX',
    'IIIIIIIIIIIIIIIYZZZXXY', 'IIIIIIIIIIIIIIIYZZZYYY',
    'IIIIIIIIIIIIIIIXZZZXYY', 'IIIIIIIIIIIIIIIYZZZYXX',
    'IIIIIIIIIIIIIIIXZZZXXX', 'IIIIIIIIIIIIIIIXZZZYYX',
    'IIIIIIIIIIIIIIYZZZZXXY', 'IIIIIIIIIIIIIIYZZZZYYY',
    'IIIIIIIIIIIIIIXZZZZXYY', 'IIIIIIIIIIIIIIYZZZZYXX',
    'IIIIIIIIIIIIIIXZZZZXXX', 'IIIIIIIIIIIIIIXZZZZYYX',
    'IIIIIIIIIIIIIYZZZZZXXY', 'IIIIIIIIIIIIIYZZZZZYYY',
    'IIIIIIIIIIIIIXZZZZZXYY', 'IIIIIIIIIIIIIYZZZZZYXX',
    'IIIIIIIIIIIIIXZZZZZXXX', 'IIIIIIIIIIIIIXZZZZZYYX',
    'IIIIIIIIIIIIYZZZZZZXXY', 'IIIIIIIIIIIIYZZZZZZYYY',
    'IIIIIIIIIIIIXZZZZZZXYY', 'IIIIIIIIIIIIYZZZZZZYXX',
    'IIIIIIIIIIIIXZZZZZZXXX', 'IIIIIIIIIIIIXZZZZZZYYX',
    'IIIIIIIIIIIYZZZZZZZXXY', 'IIIIIIIIIIIYZZZZZZZYYY',
    'IIIIIIIIIIIXZZZZZZZXYY', 'IIIIIIIIIIIYZZZZZZZYXX',
    'IIIIIIIIIIIXZZZZZZZXXX', 'IIIIIIIIIIIXZZZZZZZYYX',
    'IIIIIIIIIIIIIIIIIYXIXY', 'IIIIIIIIIIIIIIIIIYYIYY',
    'IIIIIIIIIIIIIIIIIXXIYY', 'IIIIIIIIIIIIIIIIIYYIXX',
    'IIIIIIIIIIIIIIIIIXXIXX', 'IIIIIIIIIIIIIIIIIXYIYX',
    'IIIIIIIIIIIIIIIIYZXIXY', 'IIIIIIIIIIIIIIIIYZYIYY',
    'IIIIIIIIIIIIIIIIXZXIYY', 'IIIIIIIIIIIIIIIIYZYIXX',
    'IIIIIIIIIIIIIIIIXZXIXX', 'IIIIIIIIIIIIIIIIXZYIYX',
    'IIIIIIIIIIIIIIIYZZXIXY', 'IIIIIIIIIIIIIIIYZZYIYY',
    'IIIIIIIIIIIIIIIXZZXIYY', 'IIIIIIIIIIIIIIIYZZYIXX',
    'IIIIIIIIIIIIIIIXZZXIXX', 'IIIIIIIIIIIIIIIXZZYIYX',
    'IIIIIIIIIIIIIIYZZZXIXY', 'IIIIIIIIIIIIIIYZZZYIYY',
    'IIIIIIIIIIIIIIXZZZXIYY', 'IIIIIIIIIIIIIIYZZZYIXX',
    'IIIIIIIIIIIIIIXZZZXIXX', 'IIIIIIIIIIIIIIXZZZYIYX',
    'IIIIIIIIIIIIIYZZZZXIXY', 'IIIIIIIIIIIIIYZZZZYIYY',
    'IIIIIIIIIIIIIXZZZZXIYY', 'IIIIIIIIIIIIIYZZZZYIXX',
    'IIIIIIIIIIIIIXZZZZXIXX', 'IIIIIIIIIIIIIXZZZZYIYX',
    'IIIIIIIIIIIIYZZZZZXIXY', 'IIIIIIIIIIIIYZZZZZYIYY',
    'IIIIIIIIIIIIXZZZZZXIYY', 'IIIIIIIIIIIIYZZZZZYIXX',
    'IIIIIIIIIIIIXZZZZZXIXX', 'IIIIIIIIIIIIXZZZZZYIYX',
    'IIIIIIIIIIIYZZZZZZXIXY', 'IIIIIIIIIIIYZZZZZZYIYY',
    'IIIIIIIIIIIXZZZZZZXIYY', 'IIIIIIIIIIIYZZZZZZYIXX',
    'IIIIIIIIIIIXZZZZZZXIXX', 'IIIIIIIIIIIXZZZZZZYIYX',
    'IIIIIIIIIIIIIIIIYXIIXY', 'IIIIIIIIIIIIIIIIYYIIYY',
    'IIIIIIIIIIIIIIIIXXIIYY', 'IIIIIIIIIIIIIIIIYYIIXX',
    'IIIIIIIIIIIIIIIIXXIIXX', 'IIIIIIIIIIIIIIIIXYIIYX',
    'IIIIIIIIIIIIIIIYZXIIXY', 'IIIIIIIIIIIIIIIYZYIIYY',
    'IIIIIIIIIIIIIIIXZXIIYY', 'IIIIIIIIIIIIIIIYZYIIXX'])

coeffs = np.array([
    -2.38476799e-06+0.j, -2.54069063e-06+0.j, -1.55922634e-07+0.j,
    -1.55922634e-07+0.j, -2.54069063e-06+0.j, -2.38476799e-06+0.j,
    -3.25786104e-06+0.j, -7.12962163e-06+0.j, -3.87176059e-06+0.j,
    -3.87176059e-06+0.j, -7.12962163e-06+0.j, -3.25786104e-06+0.j,
    -1.34019018e-04+0.j, -1.74138457e-04+0.j, -4.01194385e-05+0.j,
    -4.01194385e-05+0.j, -1.74138457e-04+0.j, -1.34019018e-04+0.j,
    4.94958014e-05+0.j,  6.41626617e-05+0.j,  1.46668603e-05+0.j,
    1.46668603e-05+0.j,  6.41626617e-05+0.j,  4.94958014e-05+0.j,
    8.55602904e-05+0.j,  9.18732766e-05+0.j,  6.31298618e-06+0.j,
    6.31298618e-06+0.j,  9.18732766e-05+0.j,  8.55602904e-05+0.j,
    -7.31341568e-03+0.j, -7.63751298e-03+0.j, -3.24097301e-04+0.j,
    -3.24097301e-04+0.j, -7.63751298e-03+0.j, -7.31341568e-03+0.j,
    -5.83847754e-05+0.j, -6.64310595e-05+0.j, -8.04628410e-06+0.j,
    -8.04628410e-06+0.j, -6.64310595e-05+0.j, -5.83847754e-05+0.j,
    9.48252613e-05+0.j,  1.17580843e-04+0.j,  2.27555813e-05+0.j,
    2.27555813e-05+0.j,  1.17580843e-04+0.j,  9.48252613e-05+0.j,
    1.93016760e-05+0.j,  2.27560060e-05+0.j,  3.45432999e-06+0.j,
    3.45432999e-06+0.j,  2.27560060e-05+0.j,  1.93016760e-05+0.j,
    -2.27329986e-06+0.j,  5.24808543e-06+0.j,  7.52138529e-06+0.j,
    7.52138529e-06+0.j,  5.24808543e-06+0.j, -2.27329986e-06+0.j,
    -1.43603748e-05+0.j, -2.37960262e-05+0.j, -9.43565140e-06+0.j,
    -9.43565140e-06+0.j, -2.37960262e-05+0.j, -1.43603748e-05+0.j,
    2.89120785e-05+0.j,  3.74584947e-05+0.j,  8.54641621e-06+0.j,
    8.54641621e-06+0.j,  3.74584947e-05+0.j,  2.89120785e-05+0.j,
    -1.64124583e-03+0.j, -1.83869099e-03+0.j, -1.97445158e-04+0.j,
    -1.97445158e-04+0.j, -1.83869099e-03+0.j, -1.64124583e-03+0.j,
    -6.22799134e-06+0.j,  3.67293903e-06+0.j,  9.90093037e-06+0.j,
    9.90093037e-06+0.j,  3.67293903e-06+0.j, -6.22799134e-06+0.j,
    1.10777376e-05+0.j,  8.53136587e-06+0.j, -2.54637170e-06+0.j,
    -2.54637170e-06+0.j,  8.53136587e-06+0.j,  1.10777376e-05+0.j,
    -1.10711295e-05+0.j,  6.60760260e-05+0.j,  7.71471555e-05+0.j,
    7.71471555e-05+0.j,  6.60760260e-05+0.j, -1.10711295e-05+0.j,
    -9.61269282e-06+0.j, -3.07028288e-05+0.j, -2.10901360e-05+0.j,
    -2.10901360e-05+0.j
])


spop = SparsePauliOp(oplist, coeffs = coeffs)
op = oplist[0]
coeff = coeffs[0]

print (("op=%s\nz=%s\nx=%s\nphase=%s\n" % (op, op.z, op.x, op._phase[0])))

timer("to_matrix", lambda: op.to_matrix(sparse=True))
timer("to_matrix(accel=False)", lambda: op.to_matrix(sparse=True, accel=False))

timer("_to_matrix", lambda: BasePauli._to_matrix(op.z, op.x, op._phase[0], sparse=True))

timer("_to_matrix0", lambda: BasePauli._to_matrix0(op.z, op.x, op._phase[0], sparse=True))

mat = timer("sparse_to_matrix", lambda: sparse_to_matrix(op.z, op.x, coeff=coeff, phase=op._phase[0]))
print(repr(mat))

def doit():
    coeff = 1.0 + 0j
    for i in range(len(oplist)):
        _ = oplist[i].to_matrix(coeff=coeff, sparse=True)

def doit_slow_coeff():
    for i in range(len(oplist)):
        _ = coeffs[i] * oplist[i].to_matrix(sparse=True)

def doit_fast_coeff():
    for i in range(len(oplist)):
        _ = oplist[i].to_matrix(coeff=coeffs[i], sparse=True)

timer("doit", doit)
timer("doit_slow_coeff", doit_slow_coeff)
timer("doit_fast_coeff", doit_fast_coeff)

timer("spop.to_matrix(sparse=True)", lambda: spop.to_matrix(sparse=True))

import timeit


print(timeit.timeit(lambda: coeff * mat, number=1000))

print ("args: %s" % ((op.z, op.x, op._phase[0]),))
print(timeit.timeit(lambda: sparse_to_matrix(op.z, op.x, coeff=coeff, phase=op._phase[0]), number=10))

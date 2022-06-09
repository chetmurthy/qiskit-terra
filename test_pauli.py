
from qiskit.quantum_info.operators.symplectic.pauli import *
import qiskit._accelerate
import numpy as np

def to_matrix_orig(z, x, phase=0, group_phase=False):
    num_qubits = z.size

    print(("1: z=%s x=%s num_qubits=%s mut_phase=%s" % (z, x, num_qubits, phase)))

    # Convert to zx_phase
    if group_phase:
        phase += np.sum(x & z)
        phase %= 4

    print(("2: mut_phase=%s" % (phase,)))

    dim = 2**num_qubits
    twos_array = 1 << np.arange(num_qubits)
    print(("3: twos_array=%s" % (twos_array,)))
    x_indices = np.asarray(x).dot(twos_array)
    z_indices = np.asarray(z).dot(twos_array)
    print(("4: x_indices=%s z_indices=%s" % (x_indices,z_indices)))

    indptr = np.arange(dim + 1, dtype=np.uint)
    indices = indptr ^ x_indices
    if phase:
        coeff = (-1j) ** phase
    else:
        coeff = 1
    #data = np.array([coeff * (-1) ** (bin(i).count("1") % 2) for i in z_indices & indptr])
    data = make_data(coeff, z_indices, indptr)
    # Return sparse matrix
    from scipy.sparse import csr_matrix

    #return csr_matrix((data, indices, indptr), shape=(dim, dim), dtype=complex)
    return (data,indices, indptr)

def make_data0(coeff, z_indices, indptr):
    data = np.array([coeff * (-1) ** (bin(i).count("1") % 2) for i in z_indices & indptr])
    return data

def make_data(coeff, z_indices, indptr):
    data = np.full(indptr.shape, coeff, dtype=np.complex128)
    negatives = np.bitwise_xor.reduce(np.unpackbits((z_indices & indptr)[:, None].view(np.uint8), axis=1), axis=1) == 1
    data[negatives] *= -1
    return data

coeff = -1j
z = np.array([True, True, True, True], dtype=bool)
x = np.array([True, True, True, True], dtype=bool)
phase = 0
group_phase = False
op = Pauli('Y' * 4)

if __name__ == "__main__":
    qiskit._accelerate.base_pauli.make_data(z,x,phase,group_phase)


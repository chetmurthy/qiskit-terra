# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Eigensolver algorithm."""

import logging
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse as scisparse

from qiskit.opflow import I, ListOp, OperatorBase, StateFn
from qiskit.utils.validation import validate_min
from ..exceptions import AlgorithmError
from .eigen_solver import Eigensolver, EigensolverResult
from .numpy_eigen_solver import NumPyEigensolver
from ..list_or_dict import ListOrDict

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name


class DavidsonEigensolver(NumPyEigensolver):
    r"""
    The Davidson Eigensolver algorithm.

    Davidson Eigensolver computes up to the first :math:`k` eigenvalues of a complex-valued square
    matrix of dimension :math:`n \times n`, with :math:`k \leq n`.

    Note:
        Operators are automatically converted to SciPy's ``spmatrix``
        as needed and this conversion can be costly in terms of memory and performance as the
        operator size, mostly in terms of number of qubits it represents, gets larger.
    """

    def __init__(
        self,
        k: int = 1,
        filter_criterion: Callable[
            [Union[List, np.ndarray], float, Optional[ListOrDict[float]]], bool
        ] = None,
        initial_guess: np.ndarray = None,
        preconditioner: Callable[
            [np.ndarray, complex, np.ndarray], np.ndarray
        ] = None,
    ) -> None:
        """
        Args:
            k: How many eigenvalues are to be computed, has a min. value of 1.
            filter_criterion: callable that allows to filter eigenvalues/eigenstates, only feasible
                eigenstates are returned in the results. The callable has the signature
                `filter(eigenstate, eigenvalue, aux_values)` and must return a boolean to indicate
                whether to keep this value in the final returned result or not. If the number of
                elements that satisfies the criterion is smaller than `k` then the returned list has
                fewer elements and can even be empty.
            initial_guess: best to look at the PySCF documentation for this
            preconditioner: best to look at the PySCF documentation for this
        """
        if k is None:
            raise AlgorithmError("Davidson eigensolver is only useful for computing lowest eigenvalues; for all eigenvalues, use standard eigensolver")
        super().__init__(k, filter_criterion)
        self._initial_guess = initial_guess
        self._preconditioner = preconditioner

    def _solve(self, operator: OperatorBase) -> None:
        #if self._k != 1: return super()._solve(operator)
        sp_mat = operator.to_spmatrix()
        # If matrix is diagonal, the elements on the diagonal are the eigenvalues. Solve by sorting.
        if scisparse.csr_matrix(sp_mat.diagonal()).nnz == sp_mat.nnz:
            diag = sp_mat.diagonal()
            indices = np.argsort(diag)[: self._k]
            eigval = diag[indices]
            eigvec = np.zeros((sp_mat.shape[0], self._k))
            for i, idx in enumerate(indices):
                eigvec[idx, i] = 1.0
        else:
            eigval, eigvec = sparse_davidson_eig(
                sp_mat,
                nroots=self._k,
                initial_guess = self._initial_guess,
                preconditioner = self._preconditioner
            )
            if eigval.size != self._k:
                raise AlgorithmError(("wanted %d roots but only found %d" % (self._k, eigval.size)))
            indices = np.argsort(eigval)[: self._k]
            eigval = eigval[indices]
            eigvec = eigvec[:, indices]
        self._ret.eigenvalues = eigval
        self._ret.eigenstates = eigvec.T


def sparse_initial_guess(H):
    x = np.zeros(H.shape[0])
    x[np.argmin(H.diagonal())] = 1
    return x

from pyscf import lib

def sparse_davidson_eig(sp_mat,tol=1e-14, nroots = 1,
                        initial_guess = None, preconditioner = None):
    def reg(x,tol):
        def f(x):
            if(np.abs(x)<tol): return tol
            return x
        f = np.vectorize(f)
        return f(x)

    aop = lambda x: sp_mat.dot(x)
    def precond(dx,e,x0):
        rv = dx/reg(sp_mat.diagonal()-e,tol)
        return rv
    if initial_guess is None: initial_guess = sparse_initial_guess(sp_mat)
    if preconditioner is None: preconditioner = precond
    shape = sp_mat.shape
    e, c = lib.davidson(aop, initial_guess, preconditioner, nroots=nroots)
    if nroots == 1:
        e = np.array([e])
        c = np.array([c])
    else:
        e = np.array(e)
        c = np.array(c)
    c = c.T
    return (e,c)

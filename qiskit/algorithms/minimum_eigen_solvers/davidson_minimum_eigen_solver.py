# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Davidson Minimum Eigensolver algorithm."""

from typing import List, Optional, Union, Callable
import logging
import numpy as np

from qiskit.opflow import OperatorBase
from ..eigen_solvers.davidson_eigen_solver import DavidsonEigensolver
from .minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult
from ..list_or_dict import ListOrDict

logger = logging.getLogger(__name__)


class DavidsonMinimumEigensolver(MinimumEigensolver):
    """
    The Davidson Minimum Eigensolver algorithm.
    """

    def __init__(
        self,
        initial_guess: np.ndarray = None,
        preconditioner: Callable[
            [np.ndarray, complex, np.ndarray], np.ndarray
        ] = None,
    ) -> None:
        """
        Args:
            initial_guess: best to look at the PySCF documentation for this
            preconditioner: best to look at the PySCF documentation for this
        """
        self._ces = DavidsonEigensolver(initial_guess = initial_guess,
                                        preconditioner = preconditioner
                                        )
        self._ret = MinimumEigensolverResult()

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return DavidsonEigensolver.supports_aux_operators()

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[ListOrDict[OperatorBase]] = None
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)
        result_ces = self._ces.compute_eigenvalues(operator, aux_operators)
        self._ret = MinimumEigensolverResult()
        if result_ces.eigenvalues is not None and len(result_ces.eigenvalues) > 0:
            self._ret.eigenvalue = result_ces.eigenvalues[0]
            self._ret.eigenstate = result_ces.eigenstates[0]
            if result_ces.aux_operator_eigenvalues:
                self._ret.aux_operator_eigenvalues = result_ces.aux_operator_eigenvalues[0]

        logger.debug("MinimumEigensolver:\n%s", self._ret)

        return self._ret

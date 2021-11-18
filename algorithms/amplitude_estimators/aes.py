# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Maximum Likelihood Amplitude Estimation algorithm."""

from numbers import Number
from typing import Optional, List, Union, Tuple, Dict, Callable
import numpy as np
from scipy.optimize import brute
from scipy.stats import norm, chi2

from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.utils import QuantumInstance
from qiskit.extensions import UnitaryGate

from .amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from .estimation_problem import EstimationProblem
from ..exceptions import AlgorithmError

MINIMIZER = Callable[[Callable[[float], float], List[Tuple[float, float]]], float]


class AmplitudeEstimationSimplified(AmplitudeEstimator):
    """The Amplitude Estimation Simplified algorithm.

    This class implements the quantum amplitude estimation (QAE) algorithm without phase
    estimation, as introduced in [1]. In comparison to the original QAE algorithm [2],
    this implementation relies solely on different powers of the Grover operator and does not
    require additional evaluation qubits.

    References:
        [1]: .
        [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(
        self,
        epsilon: float, 
        delta: float,
        marked: List[Number],
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        r"""
        Args:
            epsilon: The desired accuracy of our estimation. Given a true amplitude a, the algorithm 
                returns a* where 
                    a(1-epsilon) < a* < a(1+epsilon)
            delta: The probability that the true value is outside of the final confidence interval 
            marked: The list of marked elements for this instance
            quantum_instance: Quantum Instance or Backend

        Raises:
            ValueError: If the number of oracle circuits is smaller than 1.
        """
        # validate ranges of input arguments
        if not 0 < epsilon <= 0.5:
            raise ValueError(f"The target epsilon must be in (0, 0.5], but is {epsilon}.")

        if not 0 < delta < 1:
            raise ValueError(f"The confidence level delta must be in (0, 1), but is {delta}")

        super().__init__()

        # set quantum instance
        self.quantum_instance = quantum_instance
        self._marked = marked
        self._epsilon = epsilon 
        self._delta = delta 

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Get the quantum instance.

        Returns:
            The quantum instance used to run this algorithm.
        """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, BaseBackend, Backend]
    ) -> None:
        """Set quantum instance.

        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    def construct_circuit(
        self, estimation_problem: EstimationProblem, k: int, measurement: bool = False
    ) -> Union[QuantumCircuit, Tuple[QuantumCircuit, List[int]]]:
        r"""Construct the circuit :math:`Q^k A |00\rangle>`.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            estimation_problem: The estimation problem for which to construct the circuit.
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit :math:`Q^k A |0\rangle`.
        """
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )

        # This algorithm requires 2 ancilla qubits
        circuit = QuantumCircuit(num_qubits+2)
        A_ = estimation_problem.state_preparation

        # Define A operator 
        A_qc = QuantumCircuit(num_qubits+2)
        A_qc.append(A_, range(num_qubits))
        marked_s = sorted(self._marked, reverse=True)
        for target in range(16):
            if marked_s and marked_s[-1] == target:
                marked_s.pop()
            else:
                if not target & 0b1000:
                    A_qc.x(3)
                if not target & 0b0100:
                    A_qc.x(2)
                if not target & 0b0010:
                    A_qc.x(1)
                if not target & 0b0001:
                    A_qc.x(0)
                A_qc.mcx([0,1,2,3], 4)
                if not target & 0b0001:
                    A_qc.x(0)
                if not target & 0b0010:
                    A_qc.x(1)
                if not target & 0b0100:
                    A_qc.x(2)
                if not target & 0b1000:
                    A_qc.x(3)
        A_gate = A_qc.to_gate(label="A")
        
        # Define A x R operator 
        AR_qc = QuantumCircuit(num_qubits+2)
        AR_qc.append(A_, range(num_qubits))
        marked_s = sorted(self._marked, reverse=True)
        for target in range(16):
            if marked_s and marked_s[-1] == target:
                marked_s.pop()
            else:
                if not target & 0b1000:
                    AR_qc.x(3)
                if not target & 0b0100:
                    AR_qc.x(2)
                if not target & 0b0010:
                    AR_qc.x(1)
                if not target & 0b0001:
                    AR_qc.x(0)
                AR_qc.mcx([0,1,2,3], 4)
                if not target & 0b0001:
                    AR_qc.x(0)
                if not target & 0b0010:
                    AR_qc.x(1)
                if not target & 0b0100:
                    AR_qc.x(2)
                if not target & 0b1000:
                    AR_qc.x(3)
        AR_qc.u(3.13959465126, 0, 0, num_qubits+1)  # R 
        AR_gate = AR_qc.to_gate(label="A x R")

        # Define inverse operator 
        AR_gate_inv = AR_gate.inverse()

        # Define the two other gates used in the Grover diffusion operator 
        zero_vec = np.zeros(2**(num_qubits+2))
        zero_vec[0] = 1
        zero_gate = UnitaryGate(np.eye(2**(num_qubits+2)) - 2 * np.tensordot(zero_vec, zero_vec, axes=0))

        measure_vec = np.zeros(2**2)
        measure_vec[0] = 1
        measure_gate = UnitaryGate(-1 * (np.eye(2**(num_qubits+2)) - 2 * (np.tensordot(np.eye(2**num_qubits), np.tensordot(measure_vec, measure_vec, axes=0), axes=0)).swapaxes(1, 2).reshape(2**(num_qubits+2), 2**(num_qubits+2))))

        # Define the grover operator 
        G_qc = QuantumCircuit(num_qubits+2)
        G_qc.append(measure_gate, range(6))
        G_qc.append(AR_gate_inv, range(6))
        G_qc.append(zero_gate, range(6))
        G_qc.append(AR_gate, range(6))
        G_gate = G_qc.to_gate(label="G")

        # add classical register if needed
        # In this algorithm, we only need to measure the two ancilla qubits
        if measurement:
            c = ClassicalRegister(2)
            circuit.add_register(c)

        # add A operator
        circuit.append(AR_gate, range(num_qubits+2))

        # add Q^k
        if k != 0:
            circuit.append(G_gate.power(k), range(num_qubits+2))

            # add optional measurement
        if measurement:
            # real hardware can currently not handle operations after measurements, which might
            # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
            circuit.barrier()
            circuit.measure(range(num_qubits, num_qubits+2), c[:])

        return circuit

    def estimate(
        self, estimation_problem: EstimationProblem
    ) -> "AmplitudeEstimationSimplifiedEstimationResult":
        if estimation_problem.state_preparation is None:
            raise AlgorithmError(
                "Either the state_preparation variable or the a_factory "
                "(deprecated) must be set to run the algorithm."
            )

        result = AmplitudeEstimationSimplifiedEstimationResult()
        # result.post_processing = estimation_problem.post_processing

        if self._quantum_instance.is_statevector:
            # run circuit on statevector simulator
            circuits = self.construct_circuits(estimation_problem, measurement=False)
            ret = self._quantum_instance.execute(circuits)

            # get statevectors and construct MLE input
            statevectors = [np.asarray(ret.get_statevector(circuit)) for circuit in circuits]
            result.circuit_results = statevectors

            # to count the number of Q-oracle calls (don't count shots)
            result.shots = 1

        else:
            k = 0 

            while True: 
                r_k = np.floor(1.05 ** k)

                circ = self.construct_circuit(estimation_problem, (r_k - 1) // 2, measurement=True)
                self._quantum_instance._run_config.shots = 5000 * np.log(5.0 / self._delta)
                ret = self._quantum_instance.execute(circ)
                print(ret.get_counts())
                success_rate = float(ret.get_counts()['00']) / self._quantum_instance._run_config.shots
                print(k, r_k, '00:', success_rate)
                if success_rate > 0.95:
                    break 
                k += 1
            
            theta_min = 0.9 * (1.05 ** (-k))
            theta_max = 1.65 * theta_min 

            for t in range(300):
                r_t = self.rotation(theta_min, theta_max)
                del_t = (self._delta * self._epsilon / 65.0) * (0.9 ** (-t))

                circ = self.construct_circuit(estimation_problem, (r_t - 1) // 2, measurement=True)
                self._quantum_instance._run_config.shots = 250 * np.log(1.0 / del_t)
                ret = self._quantum_instance.execute(circ)
                success_rate = float(ret.get_counts()['00']) / self._quantum_instance._run_config.shots

                gamma = theta_max / theta_min - 1
                if success_rate > 0.12:
                    theta_min = theta_max / (1.0 + 0.9 * gamma)
                else:
                    theta_max = (1.0 + 0.9 * gamma) * theta_min
                
                print(t, f"[{theta_max}, {theta_min}]")
                if theta_max <= (1 + self._epsilon / 5) * theta_min:
                    print("breaking..")
                    break 
            
            return 1001 * np.sin(theta_max) 
            # to count the number of Q-oracle calls
            result.shots = self._quantum_instance._run_config.shots

        # run maximum likelihood estimation
        num_state_qubits = circuits[0].num_qubits - circuits[0].num_ancillas
        theta, good_counts = self.compute_mle(
            result.circuit_results, estimation_problem, num_state_qubits, True
        )

        # store results
        result.theta = theta
        result.good_counts = good_counts
        result.estimation = np.sin(result.theta) ** 2

        # not sure why pylint complains, this is a callable and the tests pass
        # pylint: disable=not-callable
        result.estimation_processed = result.post_processing(result.estimation)

        result.fisher_information = _compute_fisher_information(result)
        result.num_oracle_queries = result.shots * sum(k for k in result.evaluation_schedule)

        # compute and store confidence interval
        confidence_interval = self.compute_confidence_interval(result, alpha=0.05, kind="fisher")
        result.confidence_interval = confidence_interval
        result.confidence_interval_processed = tuple(
            estimation_problem.post_processing(value) for value in confidence_interval
        )

        return result


    def rotation(self, theta_min, theta_max):
        """
        Implements Lemma 2 
        """

        dtheta = theta_max - theta_min
        k = np.round(theta_min / (2.0 * dtheta))
        return 2 * np.floor( np.pi * k / theta_min ) + 1



class AmplitudeEstimationSimplifiedEstimationResult(AmplitudeEstimatorResult):
    """The ``AmplitudeEstimationSimplifiedEstimationResult`` result object."""

    def __init__(self) -> None:
        super().__init__()
        self._theta = None
        self._minimizer = None
        self._good_counts = None
        self._evaluation_schedule = None
        self._fisher_information = None

    @property
    def theta(self) -> float:
        r"""Return the estimate for the angle :math:`\theta`."""
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        r"""Set the estimate for the angle :math:`\theta`."""
        self._theta = value

    @property
    def minimizer(self) -> callable:
        """Return the minimizer used for the search of the likelihood function."""
        return self._minimizer

    @minimizer.setter
    def minimizer(self, value: callable) -> None:
        """Set the number minimizer used for the search of the likelihood function."""
        self._minimizer = value

    @property
    def good_counts(self) -> List[float]:
        """Return the percentage of good counts per circuit power."""
        return self._good_counts

    @good_counts.setter
    def good_counts(self, counts: List[float]) -> None:
        """Set the percentage of good counts per circuit power."""
        self._good_counts = counts

    @property
    def evaluation_schedule(self) -> List[int]:
        """Return the evaluation schedule for the powers of the Grover operator."""
        return self._evaluation_schedule

    @evaluation_schedule.setter
    def evaluation_schedule(self, evaluation_schedule: List[int]) -> None:
        """Set the evaluation schedule for the powers of the Grover operator."""
        self._evaluation_schedule = evaluation_schedule

    @property
    def fisher_information(self) -> float:
        """Return the Fisher information for the estimated amplitude."""
        return self._fisher_information

    @fisher_information.setter
    def fisher_information(self, value: float) -> None:
        """Set the Fisher information for the estimated amplitude."""
        self._fisher_information = value


def _safe_min(array, default=0):
    if len(array) == 0:
        return default
    return np.min(array)


def _safe_max(array, default=(np.pi / 2)):
    if len(array) == 0:
        return default
    return np.max(array)


def _get_counts(
    circuit_results: List[Union[np.ndarray, List[float], Dict[str, int]]],
    estimation_problem: EstimationProblem,
    num_state_qubits: int,
) -> Tuple[List[float], List[int]]:
    """Get the good and total counts.

    Returns:
        A pair of two lists, ([1-counts per experiment], [shots per experiment]).

    Raises:
        AlgorithmError: If self.run() has not been called yet.
    """
    one_hits = []  # h_k: how often 1 has been measured, for a power Q^(m_k)
    all_hits = []  # shots_k: how often has been measured at a power Q^(m_k)
    if all(isinstance(data, (list, np.ndarray)) for data in circuit_results):
        probabilities = []
        num_qubits = int(np.log2(len(circuit_results[0])))  # the total number of qubits
        for statevector in circuit_results:
            p_k = 0.0
            for i, amplitude in enumerate(statevector):
                probability = np.abs(amplitude) ** 2
                # consider only state qubits and revert bit order
                bitstr = bin(i)[2:].zfill(num_qubits)[-num_state_qubits:][::-1]
                objectives = [bitstr[index] for index in estimation_problem.objective_qubits]
                if estimation_problem.is_good_state(objectives):
                    p_k += probability
            probabilities += [p_k]

        one_hits = probabilities
        all_hits = np.ones_like(one_hits)
    else:
        for counts in circuit_results:
            all_hits.append(sum(counts.values()))
            one_hits.append(
                sum(
                    count
                    for bitstr, count in counts.items()
                    if estimation_problem.is_good_state(bitstr)
                )
            )

    return one_hits, all_hits

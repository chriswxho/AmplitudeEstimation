from ast import Num
import matplotlib.pyplot as plt
import numpy as np

from random import sample, seed

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble
from qiskit.quantum_info import Statevector
from qiskit.algorithms import amplitude_estimators, EstimationProblem
from qiskit.extensions import UnitaryGate
from qiskit.utils import QuantumInstance

from qiskit.visualization import plot_histogram

from algorithms.amplitude_estimators.aes import AmplitudeEstimationSimplified as AES
from operators import *

n = 4
N = 2**n
k = N//2
marked = sample(range(N), k)
print(marked)

# Define the estimation problem
# https://qiskit.org/documentation/stubs/qiskit.algorithms.EstimationProblem.html#qiskit.algorithms.EstimationProblem
def good_state(state):
    bin_marked = [(n-len(bin(s))+2)*'0'+bin(s)[2:] for s in marked]
    return (state in bin_marked)

problem = EstimationProblem(
    state_preparation=A(n),  # A operator
    grover_operator=Q(n, marked),  # Q operator
    objective_qubits=range(n),
    is_good_state=good_state  # the "good" state Psi1 is identified as measuring |1> in qubit 0
)

# use local simulator
aer_sim = Aer.get_backend('aer_simulator')
nshots = int(5000 * np.log(5.0 / 0.5))
aer_QI = QuantumInstance(aer_sim, shots=nshots)

print(nshots)
aes = AES(0.05, 0.05, marked=marked, quantum_instance=aer_sim)

print("Estimate is: ", aes.estimate(problem))

# circ = aes.construct_circuit(problem, 4, measurement=True)

# ret = aer_QI.execute(circ)
# print(ret.get_counts())

# for k in range(4):
#   circ = aes.construct_circuit(problem, k, measurement=True)

#   ret = aer_QI.execute(circ)
#   print(ret.get_counts())
#   # success_rate = float(ret.get_counts()['00']) / nshots
#   # print(k, success_rate)

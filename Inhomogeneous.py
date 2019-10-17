#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute
from qiskit.quantum_info import Pauli, state_fidelity, basis_state, process_fidelity
from qiskit import IBMQ
import matplotlib.pyplot as plt
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[4]:

# k denotes number of circuits to run in parallel
k=1
# Set up registers for ancilla qubit, ancilla register, and work register
q0 = QuantumRegister(1, 'q0')
q1 = QuantumRegister(k, 'q1')
q2 = QuantumRegister(k, 'q2')
c0 = ClassicalRegister(k, 'c0')
c1 = ClassicalRegister(k, 'c1')
c2 = ClassicalRegister(k, 'c2')


# In[75]:

# Initialize quantum circuit
circuit = QuantumCircuit(q0, q1, q2, c0, c1, c2)
# t value at which to solve LDE
t=.5
# Normalization constant N
N=1+2*t
# xval and yval help us determine what angle to rotate by
xval=1+t
yval=t
theta1 = np.arctan2(np.sqrt(yval), np.sqrt(xval))
theta2 = np.arctan(np.sqrt(t))
for i in range(k):
	# V on ancilla qubit
    circuit.z(q0[0])
    circuit.ry(2*theta1, q0[0])
    # Ux on work qubit
    circuit.x(q2[i])
    # controlled Vs1 and Vs2 on ancilla register 
    circuit.x(q0[0])
    circuit.cz(q0[0], q1[i])
    circuit.cry(2*theta2, q0[0], q1[i])
    # controlled U1 on work register
    circuit.cx(q1[i], q2[i])
    circuit.cz(q1[i], q2[i])
    # Hermitian conjugate of Vs1 and Bs2 on ancilla register
    circuit.cry(-2*theta2, q0[0], q1[i])
    circuit.cz(q0[0], q1[i])
    circuit.x(q0[0])
    # Hermitian conjugate of V
    circuit.ry(-2*theta1, q0[0])
    circuit.z(q0[0])
    # measure output
    circuit.measure(q0[0], c2[0])
    circuit.measure(q1[i], c1[i])
    circuit.measure(q2[i], c0[i])
circuit.draw()


# In[76]:

# set up and run quantum simulator
n=1000
provider = IBMQ.get_provider()
backend = provider.get_backend('ibmqx2')
backend_sim = BasicAer.get_backend('qasm_simulator')
result = execute(circuit, backend_sim, shots=n, memory=True).result()
counts = result.get_counts(circuit)
memory = result.get_memory(circuit)
print(counts)
ones = counts['0 0 0']
zeros = counts['0 0 1']
print(ones, zeros)
alpha = np.sqrt(zeros/n)
beta = np.sqrt(ones/n)
print((alpha*N, beta*N))

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


k=1
q0 = QuantumRegister(1, 'q0')
q1 = QuantumRegister(k, 'q1')
q2 = QuantumRegister(k, 'q2')
c0 = ClassicalRegister(k, 'c0')
c1 = ClassicalRegister(k, 'c1')
c2 = ClassicalRegister(k, 'c2')


# In[75]:


circuit = QuantumCircuit(q0, q1, q2, c0, c1, c2)
t=.5
N=1+2*t
xval=1+t
yval=t
theta1 = np.arctan2(np.sqrt(yval), np.sqrt(xval))
theta2 = np.arctan(np.sqrt(t))
print(theta1, theta2)
for i in range(k):
    circuit.z(q0[0])
    circuit.ry(2*theta1, q0[0])
    circuit.x(q2[i])
    circuit.x(q0[0])
    circuit.cz(q0[0], q1[i])
    circuit.cry(2*theta2, q0[0], q1[i])
    circuit.cx(q1[i], q2[i])
    circuit.cz(q1[i], q2[i])
    circuit.cry(-2*theta2, q0[0], q1[i])
    circuit.cz(q0[0], q1[i])
    circuit.x(q0[0])
    circuit.ry(-2*theta1, q0[0])
    circuit.z(q0[0])
    circuit.measure(q0[0], c2[0])
    circuit.measure(q1[i], c1[i])
    circuit.measure(q2[i], c0[i])
circuit.draw()


# In[76]:


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


# In[51]:


print(counts)


# In[6]:


qa = QuantumRegister(1, 'q0')
qb = QuantumRegister(k, 'q1')
qc = QuantumRegister(k, 'q2')
ca = ClassicalRegister(k, 'c0')
cb = ClassicalRegister(k, 'c1')
cc = ClassicalRegister(k, 'c2')
circuit2 = QuantumCircuit(qa, qb, qc, ca, cb, cc)
circuit2.h(qb[0])
circuit2.h(qc[0])
circuit2.measure(qa[0],cc[0])
circuit2.measure(qb[0],cb[0])
circuit2.measure(qc[0],ca[0])

n=1000
provider = IBMQ.get_provider()
backend = provider.get_backend('ibmqx2')
backend_sim = BasicAer.get_backend('qasm_simulator')
result = execute(circuit2, backend_sim, shots=n, memory=True).result()
counts = result.get_counts(circuit2)
memory = result.get_memory(circuit2)
print(counts)
circuit2.draw()


# In[48]:


circuit = QuantumCircuit(q0, q1, q2, c0, c1, c2)
t=1
N=1+t
xval=1+t
yval=t
theta1 = np.arctan2(np.sqrt(yval), np.sqrt(xval))
theta2 = np.arctan(np.sqrt(t))
print(theta1, theta2)
for i in range(k):
    circuit.z(q0[0])
    circuit.ry(2*theta1, q0[0])
    circuit.x(q2[i])
    circuit.x(q0[0])
    circuit.cz(q0[0], q1[i])
    circuit.cry(2*theta2, q0[0], q1[i])
    circuit.cx(q1[i], q2[i])
    circuit.cz(q1[i], q2[i])
    circuit.cry(-2*theta2, q0[0], q1[i])
    circuit.cz(q0[0], q1[i])
    circuit.x(q0[0])
    circuit.ry(-2*theta1, q0[0])
    circuit.z(q0[0])
    circuit.measure(q0[0], c2[0])
    circuit.measure(q1[i], c1[i])
    circuit.measure(q2[i], c0[i])
circuit.draw()


# In[47]:


n=1000
provider = IBMQ.get_provider()
backend = provider.get_backend('ibmqx2')
backend_sim = BasicAer.get_backend('qasm_simulator')
result = execute(circuit, backend_sim, shots=n, memory=True).result()
counts = result.get_counts(circuit)
memory = result.get_memory(circuit)
print(counts)


# In[ ]:





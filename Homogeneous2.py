#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


k=1
q0 = QuantumRegister(k, 'q0')
q1 = QuantumRegister(k, 'q1')
c0 = ClassicalRegister(k, 'c0')
c1 = ClassicalRegister(k, 'c1')


# In[8]:


circuit = QuantumCircuit(q0, q1, c0, c1)
t=0.2
N=1+t+t**2/2+t**3/6
xval=1+t**2/2
yval=t+t**3/6
theta = np.arctan2(np.sqrt(yval), np.sqrt(xval))
print(theta)
for i in range(k):
    circuit.z(q0[i])
    circuit.ry(2*theta,q0[i])
    circuit.h(q1[i])
    circuit.ch(q0[i],q1[i])
    circuit.ry(-2*theta,q0[i])
    circuit.z(q0[i])
    circuit.measure(q0[i], c0[i])
    circuit.measure(q1[i], c1[i])
circuit.draw()


# In[9]:


n=1000
provider = IBMQ.get_provider()
backend = provider.get_backend('ibmqx2')
backend_sim = BasicAer.get_backend('qasm_simulator')
result = execute(circuit, backend_sim, shots=n, memory=True).result()
counts = result.get_counts(circuit)
memory = result.get_memory(circuit)
zeros = np.zeros(k)
ones = np.zeros(k)
for str in memory:
    (readout, ancilla) = str.split(" ");
    for i in range(k):
        if ancilla[i] == '0':
            if readout[i] == '0':
                zeros[i] += 1
            elif readout[i] == '1':
                ones[i] += 1
print(zeros)
print(ones)
xsol = N*np.sqrt(np.mean(zeros) / n)/np.sqrt(2)
ysol = N*np.sqrt(np.mean(ones) / n)/np.sqrt(2)
print((xsol, ysol))


# In[10]:


print(1/2*np.cosh(t))
print(1/2*np.cosh(t)+1/np.sqrt(2)*np.sinh(t))


# In[11]:


print(1/2+t/np.sqrt(2)+1/4*t**2+1/np.sqrt(2)/6*t**3)
print(1/2+t**2/4)


# In[ ]:





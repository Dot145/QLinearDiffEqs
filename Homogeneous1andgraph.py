#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.aer import StatevectorSimulator
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[2]:


import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute
from qiskit.quantum_info import Pauli, state_fidelity, basis_state, 
process_fidelity
from qiskit import IBMQ
import matplotlib.pyplot as plt


# In[3]:


k=1
q0 = QuantumRegister(k, 'q0')
q1 = QuantumRegister(k, 'q1')
c0 = ClassicalRegister(k, 'c0')
c1 = ClassicalRegister(k, 'c1')


# In[16]:


circuit = QuantumCircuit(q0, q1, c0, c1)
t=1
N=t+1.0-t**2/2-t**3/6+t**4/24+t**5/120
xval=1-t**2/2+t**4/24
xvalneg = False
if(xval &lt; 0):
    xval = np.abs(xval)
    xvalneg = True
yval=t-t**3/6+t**5/120
yvalneg = False
if(yval &lt; 0):
    yval = np.abs(yval)
    yvalneg = True
theta=np.arctan2(np.sqrt(yval),np.sqrt(xval))
for i in range(k):
    circuit.z(q0[i])
    circuit.ry(2*theta,q0[i])
    circuit.x(q1[i])
    circuit.cx(q0[i], q1[i])
    circuit.cz(q0[i], q1[i])
    circuit.ry(-2*theta, q0[i])
    circuit.z(q0[i])
    circuit.measure(q0[i], c0[i])
    circuit.measure(q1[i], c1[i])
circuit.draw()


# In[30]:


def homogeneous2(t):
    circuit = QuantumCircuit(q0, q1, c1)
    N=np.sqrt((1.0-t**2/2+t**4/24)**2+(1-t**3/6+t**5/120)**2)
    xval=1-t**2/2+t**4/24
    xvalneg = False
    if(xval &lt; 0):
        xval = np.abs(xval)
        xvalneg = True
    yval=t-t**3/6+t**5/120
    yvalneg = False
    if(yval &lt; 0):
        yval = np.abs(yval)
        yvalneg = True
    theta=np.arctan2(yval,xval)
    for i in range(k):
        circuit.z(q0[i])
        circuit.ry(2*theta,q0[i])
        circuit.x(q1[i])
        circuit.cx(q0[i], q1[i])
        circuit.cz(q0[i], q1[i])
        circuit.measure(q1[i], c1[i])
    return (circuit, xvalneg, yvalneg)


# In[26]:


n=1000
provider = IBMQ.get_provider()
backend = provider.get_backend('ibmqx2')
backend_sim = BasicAer.get_backend('qasm_simulator')
result = execute(circuit, backend_sim, shots=n, memory=True).result()
counts = result.get_counts(circuit)
memory = result.get_memory(circuit)

print(counts)
print(counts['0 0'])
xsolution=N*np.sqrt(counts['1 0'] / n)
ysolution = N*np.sqrt(counts['0 0'] / n)
print((xsolution, ysolution))


# In[19]:


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
xsol = N*np.sqrt(np.mean(zeros) / n)
ysol = N*np.sqrt(np.mean(ones) / n)
if(yvalneg):
    xsol = -xsol
if(xvalneg):
    ysol = -ysol
print((xsol, ysol))


# In[7]:


provider.backends()


# In[12]:


tarray = np.arange(0,2.,0.1)
vecx = np.zeros_like(tarray)
vecy = np.zeros_like(tarray)
j=0
for t in tarray:
    circuit = QuantumCircuit(q0, q1, c0, c1)
    
N=1.0+t-t**2/2-t**3/6+t**4/24+t**5/120-t**6/720-t**7/np.math.factorial(7)

    xval=1-t**2/2+t**4/24-t**6/720
    xvalneg = False
    if(xval &lt; 0):
        xval = np.abs(xval)
        xvalneg = True
    yval=t-t**3/6+t**5/120-t**7/np.math.factorial(7)
    yvalneg = False
    if(yval &lt; 0):
        yval = np.abs(yval)
        yvalneg = True
    theta=np.arctan2(-np.sqrt(yval) if yvalneg else np.sqrt(yval), -np.sqrt(xval) if xvalneg else np.sqrt(xval))
    print("t = {:1.2f}, xval={:1.2f}, is negative: {},  yval={:1.2f}, is negative: {}".format(t, xval, booltoString(xvalneg), yval, booltoString(yvalneg)))
    for i in range(k):
        circuit.z(q0[i])
        circuit.ry(2*theta,q0[i])
        circuit.x(q1[i])
        circuit.cx(q0[i], q1[i])
        circuit.cz(q0[i], q1[i])
        circuit.ry(2*theta, q0[i])
        circuit.z(q0[i])
        circuit.measure(q0[i], c0[i])
        circuit.measure(q1[i], c1[i])
    n=1000
    circuit.draw()
    backend_sim = BasicAer.get_backend('qasm_simulator')
    result = execute(circuit, backend_sim, shots=n, 
memory=True).result()
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
    xsol = N*np.sqrt(np.mean(zeros) / n)
    ysol = N*np.sqrt(np.mean(ones) / n)
    if(yvalneg):
        xsol = -xsol
    if(xvalneg):
        ysol = -ysol
    vecx[j] = xsol
    vecy[j] = ysol
    j=j+1


# In[13]:


plt.plot(tarray,vecx, 'r.', tarray, sinExpansion(tarray,3), 'r-')
plt.plot(tarray,vecy, 'b.', tarray, cosExpansion(tarray,3), 'b-')


# In[20]:


def sinExpansion(array, n):
    val = np.zeros_like(array)
    for i in range(n):
        val = val + (-1)**i*array**(2*i+1)/np.math.factorial(2*i+1)
    return val
def cosExpansion(array, n):
    val = np.zeros_like(array)
    for i in range(n):
        val = val + (-1)**i*array**(2*i)/np.math.factorial(2*i)
    return val


# In[21]:


plt.plot(tarray, sinExpansion(tarray, 3), tarray, np.sin(tarray))
plt.plot(tarray, cosExpansion(tarray, 3), tarray, np.cos(tarray))


# In[47]:


tarray=np.arange(0,2,0.1)


# In[54]:


np.arctan2(1,-1)


# In[60]:


tarray=np.arange(0,3,.1)
xval=1-tarray**2/2+tarray**4/24
yval=tarray-tarray**3/6+tarray**5/120
plt.plot(tarray,np.arctan2(yval,xval),tarray,xval,tarray,yval)


# In[10]:


def booltoString(b):
    if(b):
        return "True"
    else:
        return "False"


# In[39]:


tarray = np.arange(0,2,0.1)
vecx = np.zeros_like(tarray)
vecy = np.zeros_like(tarray)
j=0
for t in tarray:
    (circuit, xvalneg, yvalneg) = homogeneous2(t)
    n=1000
    provider = IBMQ.get_provider()
    backend = provider.get_backend('ibmqx2')
    backend_sim = BasicAer.get_backend('qasm_simulator')
    result = execute(circuit, backend_sim, shots=n, 
memory=True).result()
    counts = result.get_counts(circuit)
    memory = result.get_memory(circuit)
    xsolution=N*np.sqrt(counts.get('0',0) / n)
    ysolution = N*np.sqrt(counts.get('1',0) / n)
    vecx[j] = xsolution
    vecy[j] = ysolution
    j=j+1
plt.plot(tarray,vecx, 'r.', label='x1')
plt.plot(tarray, sinExpansion(tarray,3), 'r-')
plt.plot(tarray,vecy, 'b.', label='x2')
plt.plot(tarray, cosExpansion(tarray,3), 'b-')
plt.legend(loc='lower left')
plt.xlabel('t')
plt.ylabel('readout')


# In[32]:


tarray = np.arange(0,2.5,0.1)
vecx = np.zeros_like(tarray)
vecy = np.zeros_like(tarray)
j=0
for t in tarray:
    (circuit, xvalneg, yvalneg) = homogeneous2(t)
    n=1000
    provider = IBMQ.get_provider()
    backend = provider.get_backend('ibmqx2')
    backend_sim = BasicAer.get_backend('qasm_simulator')
    result = execute(circuit, backend_sim, shots=n, 
memory=True).result()
    counts = result.get_counts(circuit)
    memory = result.get_memory(circuit)
    xsolution=N*np.sqrt(counts.get('0',0) / n)
    ysolution = N*np.sqrt(counts.get('1',0) / n)
    if xvalneg:
        ysolution *= -1
    if yvalneg:
        xsolution *= -1
    vecx[j] = xsolution
    vecy[j] = ysolution
    j=j+1
plt.plot(tarray,vecx, 'r.', tarray, sinExpansion(tarray,3), 'r-')
plt.plot(tarray,vecy, 'b.', tarray, cosExpansion(tarray,3), 'b-')


# In[ ]:




</body></html>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Quantum Computer Simulator
# Luis Villasenor
# lvillasen@gmail.com
# 12/16/2016 Notation compatible with IBM's Quantum Experience (http://www.research.ibm.com/quantum/)
# 12/18/2016 added compact notation to expand gates applied to sequences of qubits 
# 12/26/2016 added 'init' command to initialize qubits and a teleportation example
# 12/28/2016 added  qasm2tex.py code from I. Chuang to plot circuit (https://www.media.mit.edu/quanta/qasm2circ/)
# 1/1/2017 added more gates and more examples
# 1/3/2017 added QFT and IQFT
# 1/11/2017 added Sign flip and examples on Grover search algorithm 
# 1/11/2017 added 'N&m i,j;' command to initialize basis state x to m^x mod N 
# 1/19/2017 added 'reverse' command to reverse all bits
# 1/19/2017 added negative phase shifts in 'csk' command
# 2/3/2017 added 'plot' command to plot the probabilities of the basis states
# 2/7/2017 added 'printout' command
# 2/7/2017 added 'Inverse_P_threshold' command to printout only part of the final state (handy for large number of qbits)
# 2/10/2017 added acceleration with numba

# Usage
# python QCSim.py samples/example_YYY.txt
from __future__ import print_function
import numpy as np
import numba
import sys
import string
import random
import subprocess
import os
n_qbits =0
initial=-1;q_i=-1;q_f=-1
verbose = 0; circuit = 0; plot = 0 ; printout = 1 ; P_threshold = 0.0

def get_qbits(command):
	before, sep, after = command.rpartition(";")
	g=before.split( )[0]
	if g != 'cx' and g != 'sk' and g != 'csk'  and g != 'N&m' : 
		before1, sep1, after1 = before.rpartition(":")
		if sep1 == ':': 
			a=[int(s) for s in before1 if s.isdigit()]
			if len(a)==1:qbit_i= a[0]
			if len(a)==2:qbit_i= 10*a[0]+a[1]
			a=[int(s) for s in after1 if s.isdigit()]
			if len(a)==1:qbit_f= a[0]
			if len(a)==2:qbit_f= 10*a[0]+a[1]
		else:
			a=[int(s) for s in before if s.isdigit()]
			qbit_i = 0
			for i in range(len(a)): qbit_i += a[i]*10**(len(a)-i-1)
			qbit_f=qbit_i
		qbit_c_i = qbit_i
		qbit_c_f = qbit_f
		qbit_t_i = -1
		qbit_t_f = -1
	elif g == 'sk':
		before2, sep2, after2 = before.rpartition(",")
		before1, sep1, after1 = before2.rpartition(":")
		if sep1 == ':': 
			a=[int(s) for s in before1 if s.isdigit()]
			if len(a)==1:qbit_i= a[0]
			if len(a)==2:qbit_i= 10*a[0]+a[1]
			a=[int(s) for s in after1 if s.isdigit()]
			if len(a)==1:qbit_f= a[0]
			if len(a)==2:qbit_f= 10*a[0]+a[1]
		else:
			a=[int(s) for s in before2 if s.isdigit()]
			if len(a)==1:qbit_i= a[0]
			if len(a)==2:qbit_i= 10*a[0]+a[1]
			qbit_f=qbit_i
		a=[int(s) for s in after2 if s.isdigit()]
		k=0
		for i in range(len(a)): k += a[i]*10**(len(a)-i-1)
		qbit_c_i = qbit_i
		qbit_c_f = qbit_f
		qbit_t_i = k
		qbit_t_f = -1
	elif g == 'cx':
		before2, sep2, after2 = before.rpartition(",")
		before1, sep1, after1 = before2.rpartition(":")
		if sep1 == ':': 
			a=[int(s) for s in before1 if s.isdigit()]
			if len(a)==1:qbit_c_i= a[0]
			if len(a)==2:qbit_c_i= 10*a[0]+a[1]
			a=[int(s) for s in after1 if s.isdigit()]
			if len(a)==1:qbit_c_f= a[0]
			if len(a)==2:qbit_c_f= 10*a[0]+a[1]
		else:
			a=[int(s) for s in before2 if s.isdigit()]
			if len(a)==1:qbit_c_i= a[0]
			if len(a)==2:qbit_c_i= 10*a[0]+a[1]
			qbit_c_f=qbit_c_i
		before1, sep1, after1 = after2.rpartition(":")
		if sep1 == ':': 
			a=[int(s) for s in before1 if s.isdigit()]
			if len(a)==1:qbit_t_i= a[0]
			if len(a)==2:qbit_t_i= 10*a[0]+a[1]
			a=[int(s) for s in after1 if s.isdigit()]
			if len(a)==1:qbit_t_f= a[0]
			if len(a)==2:qbit_t_f= 10*a[0]+a[1]
		else:
			a=[int(s) for s in after2 if s.isdigit()]
			if len(a)==1:qbit= a[0]
			if len(a)==2:qbit= 10*a[0]+a[1]
			qbit_t_i=qbit
			qbit_t_f=qbit_t_i
	elif g == 'csk':
		before1, sep1, after1 = before.rpartition(":")
		if sep1 == ':': 
			sys.exit('The csk gate does not allow expansion of range of qubits')
		before2, sep2, after2 = before.rpartition(",")
		before3, sep3, after3 = before2.rpartition(",")
		a=[int(s) for s in before3 if s.isdigit()]
		if len(a)==1:qbit_c= a[0]
		if len(a)==2:qbit_c= 10*a[0]+a[1]
		a=[int(s) for s in after3 if s.isdigit()]
		if len(a)==1:qbit_t= a[0]
		if len(a)==2:qbit_t= 10*a[0]+a[1]
		a=[int(s) for s in after2 if s.isdigit()]
		k=0
		for i in range(len(a)): k += a[i]*10**(len(a)-i-1)
		qbit_c_i = qbit_c
		qbit_c_f = qbit_t
		k=int(after2)
		qbit_t_i = k
		qbit_t_f = -1
	elif g == 'N&m':
		before1, sep1, after1 = before.rpartition(":")
		before2, sep2, after2 = before.rpartition(",")
		a=[int(s) for s in before2 if s.isdigit()]
		N=0
		for i in range(len(a)): N += a[i]*10**(len(a)-i-1)
		a=[int(s) for s in after2 if s.isdigit()]
		m=0
		for i in range(len(a)): m += a[i]*10**(len(a)-i-1)
		qbit_c_i = N
		qbit_c_f = m
		qbit_t_i = -1
		qbit_t_f = -1
	return qbit_c_i,qbit_c_f,qbit_t_i,qbit_t_f

if len(sys.argv) > 1:
	file=sys.argv[1]
f = open(file,"r") #opens file with QS prSignram
List = []
for line in f:
    List.append(line)

for i in range(len(List)):
	command=List[i]
	before, sep, after = command.rpartition(";")
	if before.split() != []:
		g=before.split( )[0]		
	else: g = ''
	if g =='id' or g=='h' or g=='x' or g=='y' or g=='z' or g=='s' or g=='sdg' or g=='t' or g=='tdg' or g=='measure' or g == 'QFT' or g == 'IQFT' :
		qbit_i,qbit_f,q,q = get_qbits(command)
		n_qbits=max(n_qbits, qbit_i+1)
		n_qbits=max(n_qbits, qbit_f+1)		
	elif g == 'init' or g == 'verbose' or g == 'circuit' or g == 'plot' or g == 'printout' or g == 'Inverse_P_threshold' or g == 'N&m':
		qbit_i,qbit_f,q,q = get_qbits(command)
		if g == 'init': 
			initial = -2 # init only chosen basis states
			q_i=qbit_i
			q_f=qbit_f	
		if g == 'verbose':
			verbose = qbit_i
		if g == 'circuit':
			circuit = qbit_i
			print('circuit =',circuit)
		if g == 'plot':
			plot = qbit_i
			print('plot =',plot)
		if g == 'printout':
			printout = qbit_i
			print('printout =',printout)
		if g == 'Inverse_P_threshold':
			if qbit_i > 0: P_threshold = float(1.0/qbit_i)
			print('Inverse_P_threshold =',qbit_i)
			print('P_threshold =',P_threshold)
		if g == 'N&m':
			initial = -3
			N = float(qbit_i)
			m = float(qbit_f)
	elif g == 'sk':
		qbit_i,qbit_f,k,k1 = get_qbits(command)
		n_qbits=max(n_qbits, qbit_f+1)
		n_qbits=max(n_qbits, qbit_i+1)	
	elif g == 'cx':
		qbit_c_i,qbit_c_f,qbit_t_i,qbit_t_f = get_qbits(command)
		n_qbits=max(n_qbits, qbit_c_i+1)
		n_qbits=max(n_qbits, qbit_c_f+1)
		n_qbits=max(n_qbits, qbit_t_i+1)
		n_qbits=max(n_qbits, qbit_t_f+1)
	elif g == 'csk':
		qbit_c,qbit_t,k,k1 = get_qbits(command)
		n_qbits=max(n_qbits, qbit_c+1)
		n_qbits=max(n_qbits, qbit_t+1)

def printf(str, *args):
    print(str % args, end='')

@numba.autojit
def set_bit(value, bit):
    return value | (1<<bit)

@numba.autojit
def clear_bit(value, bit):
    return value & ~(1<<bit)

def print_state(g,n_qbits,verbose,B):
	if g != 'cx' and g != 'sk' and g != 'csk' and g != 'Sign' and g != 'QFT' and g != 'IQFT': print('Gate',g,'on qubit', qbit), 
	if verbose == 1:
		printf('  resulted in state |psi> = '),
		k1=0
		psi=''
		for k in range(2**n_qbits):
			s_i=("{:0%db}"%n_qbits).format(k)[::-1]
			if B[k] != 0: 
				k1+=1
				if k1 == 1: psi += str('({:.3f}'.format(B[k])) + ')' + '|'+s_i+'> '
				else:psi += '+ '+ str('({:.3f}'.format(B[k])) + ')' + str('|'+s_i+'> ')
		psi=string.replace(psi,'+ -', '- ')
		print(psi)
		print
	C = B
	A = B
	return A,C

# quantum basis vector |j> is mapped into 1/√N ∑(from k=0 to N−1) exp(2πijk/N)|k⟩
@numba.autojit
def DFT_j(type,N,j):
    A_k = np.zeros(N,dtype=np.complex_)
    for k in range(N):
    	A_k[k] = np.exp(type*2*np.pi*1j *  j * k/N)
    return A_k/np.sqrt(N)

@numba.autojit
def DFT(n_qbits,qbit_i,qbit_f,type,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	N1 = 2**(qbit_i) 			# qbits below QFT
	N2 = 2**(qbit_f-qbit_i+1) 	# qbits at QFT
	N3 = 2**(n_qbits-qbit_f-1) 	# qbits above QFT
	for j3 in range(N3):
		for j2 in range(N2):
			for j1 in range(N1):
				j = (j3<<qbit_f+1)+(j2<<qbit_i) + j1
				if np.absolute(A[j]) > 0:
					A2 = DFT_j(type,N2,j2)
					for jj in range(len(A2)):
						j4 = (j3<<qbit_f+1) + (jj<<qbit_i) + j1
						B[j4] += A2[jj] * A[j]
	return B

@numba.autojit
def ID(n_qbits,qbit,A):
	return A

@numba.autojit
def H(n_qbits,qbit,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=1/np.sqrt(2)*A[j]
				B[set_bit(j,qbit)]+=1/np.sqrt(2)*A[j]
			elif bit_parity == 1:
				B[clear_bit(j,qbit)]+=1/np.sqrt(2)*A[j]
				B[j]+=-1/np.sqrt(2)*A[j]
	return B

@numba.autojit
def X(n_qbits,qbit,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[set_bit(j,qbit)]+=A[j]
			if bit_parity == 1:
				B[clear_bit(j,qbit)]+=A[j]
	return B

@numba.autojit
def Y(n_qbits,qbit,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[set_bit(j,qbit)]+=1j*A[j]
			if bit_parity == 1:
				B[clear_bit(j,qbit)]+=-1j*A[j]
	return B

@numba.autojit
def Z(n_qbits,qbit,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=-A[j]
	return B

@numba.autojit
def S(n_qbits,qbit,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=1j*A[j]
	return B

@numba.autojit
def Sdg(n_qbits,qbit,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=-1j*A[j]
	return B

@numba.autojit
def T(n_qbits,qbit,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=1/np.sqrt(2)*(1+1j)*A[j]
	return B

@numba.autojit
def Tdg(n_qbits,qbit,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=1/np.sqrt(2)*(1-1j)*A[j]
	return B

@numba.autojit
def Sk(n_qbits,qbit,k,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	w = np.exp(np.pi*1j/k)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=w*A[j]
	print('Gate sk on qubit', qbit,' with k =',k),	
	return B

@numba.autojit
def CX(n_qbits,qbit_c,qbit_t,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity_c=(j>>qbit_c)%2
			bit_parity_t=(j>>qbit_t)%2
			if bit_parity_c == 0:
				B[j]+=A[j]
			else:
				if bit_parity_t== 0:
					B[set_bit(j,qbit_t)]+=A[j]
				else:
					B[clear_bit(j,qbit_t)]+=A[j]
	print('Gate cx on control qubit', qbit_c,' and target qubit',qbit_t),	
	return B

@numba.autojit
def CSk(n_qbits,qbit_c,qbit_t,k,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	w = np.exp(np.pi*1j/k)
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity_c=(j>>qbit_c)%2
			bit_parity_t=(j>>qbit_t)%2
			if bit_parity_c == 0:
				B[j]+=A[j]
			else:
				if bit_parity_t == 0:
					B[j]+=A[j]
				if bit_parity_t == 1:
					B[j]+=w*A[j]
	print('Gate csk on control qubit', qbit_c,' and target qubit',qbit_t,' with k =',k),	
	return B

@numba.autojit
def Sign(n_qbits,index,A):
	B = A
	B[index] = -A[index]
	print('Sign flip on index', index),	
	return B

@numba.autojit
def Reverse(n_qbits,A):
	B = np.zeros(2**n_qbits,dtype=np.complex_)
	for j in range(2**n_qbits):
		s_j = ("{:0%db}"%n_qbits).format(j)[::]
		s_j_rev = ("{:0%db}"%n_qbits).format(j)[::-1]
		B[int(s_j_rev,2)]=A[int(s_j,2)]
	return B

if circuit == 1:
	cmd = 'echo "# Quantum score"  > QS.qasm'
	subprocess.call(cmd, shell=True)

	Sd="\'"+str('{S}^{\\dagger}')+"\'"
	cmd = 'echo "\t 	def	Sd,0,%s"  >> QS.qasm'%(Sd)
	subprocess.call(cmd, shell=True)

	Td="\'"+str('{T}^{\\dagger}')+"\'"
	cmd = 'echo "\t 	def	Td,0,%s"  >> QS.qasm'%(Td)
	subprocess.call(cmd, shell=True)

	sk = "\'" + str('S_{\pi/2}') + "\'"
	cmd = 'echo "\t 	def	Sk1,0,%s"  >> QS.qasm'%(sk)
	subprocess.call(cmd, shell=True)

	sign = "\'" + str('Sign') + "\'"
	cmd = 'echo "\t 	def	sign,0,%s"  >> QS.qasm'%(sign)
	subprocess.call(cmd, shell=True)

	for i in range(2,n_qbits):
		s = 'S_{\pi/2^{%d}}'%(i)
		sk = "\'" + s + "\'" 
		cmd = 'echo "\t 	def	Sk%s,0,%s"  >> QS.qasm'%(str(i),sk)
		subprocess.call(cmd, shell=True)

	csk = "\'" + str('S_{\pi/2}') + "\'"
	cmd = 'echo "\t 	def	CSk1,1,%s"  >> QS.qasm'%(csk)
	subprocess.call(cmd, shell=True)

	for i in range(2,n_qbits):
		s = 'S_{\pi/2^{%d}}'%(i)
		csk = "\'" + s + "\'" 
		cmd = 'echo "\t 	def	CSk%s,1,%s"  >> QS.qasm'%(str(i),csk)
		subprocess.call(cmd, shell=True)

	for qbit in range(n_qbits):
		if qbit >= q_i and qbit <= q_f:
			cmd = 'echo "\t	qubit Q%s"  >> QS.qasm'%(qbit)
			subprocess.call(cmd, shell=True)
		else:
			cmd = 'echo "\t	qubit Q%s,0"  >> QS.qasm'%(qbit)
			subprocess.call(cmd, shell=True)

print('\nNumber of qubits: ',n_qbits)
A = np.zeros(2**n_qbits,dtype=np.complex_)
M = np.zeros(n_qbits)

# Initial state
if initial == -1: # init to |000..00>
	A[0]=1

elif initial == -2: # init given basis states to random amplitude
	for k in range(2**n_qbits):
		if k >= q_i  and k <= q_f: 
			A[k] = random.uniform(-1,1)+1j*random.uniform(-1,1)
elif initial == -3: # init to |m**k mod N>
	for k in range(2**n_qbits):
		A[k] = (m**k)%N
		#if k%m == 1: 
		#	A[k] = 1

if initial != -1:
	A_norm=0
	for k in range(2**n_qbits):
		A_norm+=np.absolute(A[k])**2
	A=A/np.sqrt(A_norm)
	C=A

printf('Initial state: |psi> = '),
if initial == -1:
	k1=0
	psi=''
	for k in range(1):
		s_i=("{:0%db}"%n_qbits).format(k)[::-1]
		if A[k] != 0: 
			k1+=1
			if k1 == 1:psi += str('({:.3f}'.format(A[k])) + ')' + '|'+s_i+'> '
			else:psi+='+ '+ str('({:.3f}'.format(A[k])) + ')' + str('|'+s_i+'> ')
	psi=string.replace(psi,'+ -', '- ')
	print(psi)
	print
else:
	k1=0
	psi=''
	for k in range(2**n_qbits):
		s_i=("{:0%db}"%n_qbits).format(k)[::-1]
		if A[k] != 0: 
			k1+=1
			if k1 == 1:psi += str('({:.3f}'.format(A[k])) + ')' + '|'+s_i+'> '
			else:psi+='+ '+ str('({:.3f}'.format(A[k])) + ')' + str('|'+s_i+'> ')
	psi=string.replace(psi,'+ -', '- ')
	print(psi)
	print

for i in range(len(List)):
	command=List[i]
	before, sep, after = List[i].rpartition(";")
	if before.split() != []:
		g=before.split( )[0]
	else: g = ''
############ 1-qubit gates
################### gate id

	if g  =='id' or g  =='h' or g =='x' or g =='y' or g =='z' or g =='s' or g =='sdg' or g =='t' or g =='tdg' or g =='measure':
		qbit_i,qbit_f,q,q = get_qbits(command)

################### gate h
		if g=='h' :
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = H(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	h Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)					
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = H(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	h Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)
					for m in range(n_qbits):
						if m != qbit: 
							cmd = "echo  '\t	nop Q%s'  >> QS.qasm"%(m)
							subprocess.call(cmd, shell=True)

################### gate x			
		if g=='x':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = X(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	X Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = X(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	X Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate y
		if g =='y':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Y(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	Y Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Y(n_qbits,qbit,A)	
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	Y Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate z					
		if g =='z':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Z(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	Z Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Z(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	Z Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate s
		if g =='s':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = S(n_qbits,qbit,A)	
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	S Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = S(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	S Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate sdg
		if g =='sdg':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Sdg(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	Sd Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Sdg(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	Sd Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate t
		if g =='t':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = T(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	T Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = T(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	T Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate tdg
		if g =='tdg':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Tdg(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	Td Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Tdg(n_qbits,qbit,A)
					A,C = print_state(g,n_qbits,verbose,B)
					cmd = "echo  '\t	Td Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)

################### gate sk
	if g =='sk':
		qbit_i,qbit_f,k,k1 = get_qbits(command)
		klSign=int(np.lSign2(k))
		cs='Sk'+str(klSign)
		if qbit_f >= qbit_i:
			for qbit in range(qbit_i,qbit_f+1):
				B = Sk(n_qbits,qbit,k,A)	
				A,C = print_state(g,n_qbits,verbose,B)
				cmd = "echo  '\t	%s Q%s'  >> QS.qasm"%(cs,qbit)
				subprocess.call(cmd, shell=True)
		elif qbit_f < qbit_i:
			for qbit in range(qbit_i,qbit_f-1,-1):
				B = Sk(n_qbits,qbit,k,A)
				A,C = print_state(g,n_qbits,verbose,B)
				cmd = "echo  '\t	%s Q%s'  >> QS.qasm"%(cs,qbit)
				subprocess.call(cmd, shell=True)

############ 2-qubit gates
################### gate cx
	if g == 'cx':
		qbit_c_i,qbit_c_f,qbit_t_i,qbit_t_f = get_qbits(command)
		if qbit_c_f > qbit_c_i and qbit_t_i == qbit_t_f: # qbit_c_f > qbit_c_i qbit_t_i == qbit_t_f
			qbit_t = qbit_t_i
			for qbit_c in range(qbit_c_i,qbit_c_f+1):
				B = CX(n_qbits,qbit_c,qbit_t,A)
				A,C = print_state(g,n_qbits,verbose,B)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				subprocess.call(cmd, shell=True)
		elif qbit_c_f < qbit_c_i and qbit_t_i == qbit_t_f: # qbit_c_f < qbit_c_i qbit_t_i == qbit_t_f
			qbit_t = qbit_t_i
			for qbit_c in range(qbit_c_i,qbit_c_f-1,-1):
				B = CX(n_qbits,qbit_c,qbit_t,A)
				A,C = print_state(g,n_qbits,verbose,B)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				subprocess.call(cmd, shell=True)
		elif qbit_c_f == qbit_c_i and qbit_t_f >= qbit_t_i: # qbit_c_f == qbit_c_i and qbit_t_f > qbit_t_i
			qbit_c = qbit_c_i
			for qbit_t in range(qbit_t_i,qbit_t_f+1):
				B = CX(n_qbits,qbit_c,qbit_t,A)
				A,C = print_state(g,n_qbits,verbose,B)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				subprocess.call(cmd, shell=True)
		elif qbit_c_f == qbit_c_i and qbit_t_i >= qbit_t_f: # qbit_c_f == qbit_c_i and qbit_t_i < qbit_t_f
			qbit_c = qbit_c_i
			for qbit_t in range(qbit_t_i,qbit_t_f-1,-1):
				B = CX(n_qbits,qbit_c,qbit_t,A)				
				A,C = print_state(g,n_qbits,verbose,B)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				subprocess.call(cmd, shell=True)

################### gate csk
	if g == 'csk':
		qbit_c,qbit_t,k,k1 = get_qbits(command)
		klog=int(np.log2(abs(k)))
		cs='CSk'+str(klog)
		B = CSk(n_qbits,qbit_c,qbit_t,k,A)	
		A,C = print_state(g,n_qbits,verbose,B)
		cmd = "echo  '\t	%s Q%s,Q%s'  >> QS.qasm"%(cs,qbit_c,qbit_t)
		subprocess.call(cmd, shell=True)

################### Flip sign
	if g =='Sign':
		indx,k,k,k = get_qbits(command)
		B = Sign(n_qbits,indx,A)	
		A,C = print_state(g,n_qbits,verbose,B)	

################### Reverse qubits
	if g == 'reverse':
		B = Reverse(n_qbits,A)
		A,C = print_state(g,n_qbits,verbose,B)

################### QFT or IQFT
	if g == 'QFT' or g == 'IQFT':
		qbit_i,qbit_f,q,q = get_qbits(command)
		if g == 'QFT': 
			print('Starting QFT from qbit',qbit_i,'to qbit',qbit_f)
			type = 1
		if g == 'IQFT': 
			print('Starting IQFT from qbit',qbit_i,'to qbit',qbit_f)
			type = -1
		B = DFT(n_qbits,qbit_i,qbit_f,type,A)
		if g == 'QFT': print('Ending QFT ..')
		elif g == 'IQFT': print('Ending IQFT ..')
		A,C = print_state(g,n_qbits,verbose,B)

################### measure
	if g == 'measure':
		if qbit_f >= qbit_i:
			for qbit in range(qbit_i,qbit_f+1):
				M[qbit] = 1
				print('Measure qubit', qbit)
				cmd = "echo  '\t 	measure Q%s'  >> QS.qasm"%(qbit)
				subprocess.call(cmd, shell=True)	 
		elif qbit_f < qbit_i:
			for qbit in range(qbit_i,qbit_f-1,-1):
				M[qbit] = 1
				print('Measure qubit', qbit)
				cmd = "echo  '\t	measure Q%s'  >> QS.qasm"%(qbit)
				subprocess.call(cmd, shell=True) 

################### result
#@numba.autojit
def RES(n_qbits,C):
	#P=[0 for i in range(int(2**np.sum(M))]
	#P = np.zeros(int(2**np.sum(M)))
	'''
	Amp=['' for i in range(int(2**np.sum(M)))]

	for i in range(2**n_qbits):
		num=0
		k=0
		for j in range(n_qbits):
			if M[j] == 1:
				num+=((i>>j)&1)*2**k
				k+=1
		P[num] += np.absolute(C[i])**2

	for i in range(2**n_qbits):
		num=0
		k=0
		for j in range(n_qbits):
			if M[j] == 1:
				num+=((i>>j)&1)*2**k
				k+=1
		s_i = ("{:0%db}"%(n_qbits)).format(i)[::-1]
		if np.absolute(C[i]) > 0.0001:
			if len(Amp[num]) == 0:
				Amp[num] = Amp[num] + str('({:.3f}'.format(C[i]/np.sqrt(P[num]))) + ')' + '|' + str(s_i) + '>'
			else:
				Amp[num] = Amp[num] + str(' + ({:.3f}'.format(C[i]/np.sqrt(P[num]))) + ')' + '|' + str(s_i) + '>'
	if  np.sum(M) >0:
		print('\nProbabilities after measurement:')
	else:
		print('\nFinal state:')

	for i in range(int(2**np.sum(M))):
		s_i = ("{:0%db}"%np.sum(M)).format(i)[::-1]
		if P[i] != 0:
			if  np.sum(M) >0:
				printf('\nP(' + str(s_i) + ') = '),
				print(P[i])
			printf('|psi> = '),
			print(Amp[i])
	'''
	printf('  Final state |psi> = '),
	k1=0
	psi=''
	for k in range(2**n_qbits):
		s_i=("{:0%db}"%n_qbits).format(k)[::-1]
		if C[k] != 0: 
			k1+=1
			if k1 == 1: psi += str('({:.3f}'.format(C[k])) + ')' + '|'+s_i+'> '
			else:psi += '+ '+ str('({:.3f}'.format(C[k])) + ')' + str('|'+s_i+'> ')
	psi=string.replace(psi,'+ -', '- ')
	print(psi)
	return

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2+x.imag**2


@numba.autojit
def RESULT(n_qbits,C,P_threshold):
	print('  Final basis states with P >', P_threshold)
	P_vec = abs2(C)
	for k in range(2**n_qbits):
		if P_vec[k] > P_threshold:
			s_i=("{:0%db}"%n_qbits).format(k)[::-1] 
			psi = 'P(|'+s_i+'>) = '+str('{:.2e}'.format(P_vec[k]))+'\t'+ str(' Amplitude: {:.2e}'.format(C[k]) )
			print(psi)
	return

@numba.autojit
def PLOT(n_qbits,C):
	y = abs2(C)
	R = 1
	if n_qbits > 20:
		R = 2**(n_qbits-20)
	y1 = np.reshape(y, (-1, R)).max(axis=1) # reshape to speedup plotting
	x = np.arange(len(y1)) * R  # create x data
	fig = plt.figure(figsize=(8,7))
	ax = fig.add_subplot(111)
	plt.title("Probabilities for All Basis States",fontsize= 20,y=1.1)
	plt.xlabel("Basis State",fontsize= 20)
	plt.ylabel("Probability",fontsize= 20)
	fig.tight_layout()
	plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
	plt.semilogx(x,y1,'bo',x,y1,'r--')
	#plt.show()
	plt.savefig('plot.png')
	return 

if printout == 1: RESULT(n_qbits,C,P_threshold)

if circuit == 1:
	subprocess.call( "python qasm2tex.py QS.qasm > circ.tex",shell=True)
	subprocess.call("latex circ.tex >/dev/null 2>&1",shell=True)
	subprocess.call("dvips -D2400 -E circ.dvi >/dev/null 2>&1",shell=True)
	cmd_exists = lambda x: any(os.access(os.path.join(path, x), os.X_OK) for path in os.environ["PATH"].split(os.pathsep))
	if cmd_exists('latex') == True:
		print('\nIf latex is installed correctly then figure circ.ps was created\n')

if plot == 1: 
	import matplotlib.pyplot as plt
	print('Plotting ..')
	PLOT(n_qbits,C)

print('Done')

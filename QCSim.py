# Quantum Computer Simulator
# Luis Villasenor
# lvillasen@gmail.com
# 12/16/2016 Notation compatible with IBM's Quantum Experience (http://www.research.ibm.com/quantum/)
# 12/18/2016 added compact notation to expand gates applied to sequences of qubits 
# 12/26/2016 added init command to initialize qubits 
# 12/28/2016 added  qasm2tex.py code from I. Chuang to plot circuit (https://www.media.mit.edu/quanta/qasm2circ/)
# 1/1/2017 added more gates and more examples
# Usage
# python QCSim.py samples/example_Bell.txt
from __future__ import print_function
import numpy as np
import sys
import string
import random
import subprocess
import os


def printf(str, *args):
    print(str % args, end='')

def set_bit(value, bit):
    return value | (1<<bit)

def clear_bit(value, bit):
    return value & ~(1<<bit)

def print_state(g,n_qbits,verbose,A,B,C):
	if g != 'cx' and g != 'sk' and g != 'csk': print('Gate',g,'on qubit', qbit), 
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
	for  k in range(2**n_qbits): C[k]=B[k]
	for  k in range(2**n_qbits): A[k]=B[k]
	return A,C

def get_qbits(command):
	before, sep, after = command.rpartition(";")

	#before1, sep1, after1 = before.split()[1].rpartition(":")
	g=before.split( )[0]
	if g != 'cx' and g != 'sk' and g != 'csk' :
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
			if len(a)==1:qbit_i= a[0]
			if len(a)==2:qbit_i= 10*a[0]+a[1]
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
		print(before)
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
		qbit_t_i = k
		qbit_t_f = -1


	
	return qbit_c_i,qbit_c_f,qbit_t_i,qbit_t_f

def ID(n_qbits,A,B):
	B = [0 for i in range(2**n_qbits)]
	for j in range(2**n_qbits):
		if A[j] != 0:
			B[j]+=A[j]
	return B

def H(n_qbits,qbit,A,B):
	B = [0 for i in range(2**n_qbits)]
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=1/np.sqrt(2)*A[j]
				B[set_bit(j,qbit)]+=1/np.sqrt(2)*A[j]
			if bit_parity == 1:
				B[clear_bit(j,qbit)]+=1/np.sqrt(2)*A[j]
				B[j]+=-1/np.sqrt(2)*A[j]
	return B

def X(n_qbits,qbit,A,B):
	B = [0 for i in range(2**n_qbits)]
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[set_bit(j,qbit)]+=A[j]
			if bit_parity == 1:
				B[clear_bit(j,qbit)]+=A[j]
	return B

def Y(n_qbits,qbit,A,B):
	B = [0 for i in range(2**n_qbits)]
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[set_bit(j,qbit)]+=1j*A[j]
			if bit_parity == 1:
				B[clear_bit(j,qbit)]+=-1j*A[j]
	return B

def Z(n_qbits,qbit,A,B):
	B = [0 for i in range(2**n_qbits)]
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=-A[j]
	return B

def S(n_qbits,qbit,A,B):
	B = [0 for i in range(2**n_qbits)]
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=1j*A[j]
	return B

def Sdg(n_qbits,qbit,A,B):
	B = [0 for i in range(2**n_qbits)]
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=-1j*A[j]
	return B

def T(n_qbits,qbit,A,B):
	B = [0 for i in range(2**n_qbits)]
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=1/np.sqrt(2)*(1+1j)*A[j]
	return B

def Tdg(n_qbits,qbit,A,B):
	B = [0 for i in range(2**n_qbits)]
	for j in range(2**n_qbits):
		if A[j] != 0:
			bit_parity=(j>>qbit)%2
			if bit_parity == 0:
				B[j]+=A[j]
			if bit_parity == 1:
				B[j]+=1/np.sqrt(2)*(1-1j)*A[j]
	return B

def Sk(n_qbits,qbit,k,A,B):
	B = [0 for i in range(2**n_qbits)]
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


def CX(n_qbits,qbit_c,qbit_t,A,B):
	B = [0 for i in range(2**n_qbits)]
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

def CSk(n_qbits,qbit_c,qbit_t,k,A,B):
	B = [0 for i in range(2**n_qbits)]
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



if len(sys.argv) > 1:
	file=sys.argv[1]
f = open(file,"r") #opens file with QS program
List = []
for line in f:
    List.append(line)
n_qbits =0
q_i=-2;q_f=-2
verbose = 0
for i in range(len(List)):
	command=List[i]
	before, sep, after = command.rpartition(";")	
	if before.split() != []:
		before1, sep1, after1 = before.split()[1].rpartition(":")
		g=before.split( )[0]
	else: g = ''
	if g == 'init' or g =='id' or g=='h' or g=='x' or g=='y' or g=='z' or g=='s' or g=='sdg' or g=='t' or g=='tdg' or g=='measure' or g == 'verbose':
		qbit_i,qbit_f,q,q = get_qbits(command)
		n_qbits=max(n_qbits, qbit_i+1)
		n_qbits=max(n_qbits, qbit_f+1)
		if g == 'init': 
			q_i=qbit_i
			q_f=qbit_f	
		if g == 'verbose':
			verbose = qbit_i
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


print('\nNumber of qubits: ',n_qbits)
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



A = [0 for i in range(2**n_qbits)]
B = [0 for i in range(2**n_qbits)]
C = [0 for i in range(2**n_qbits)]
M = [0 for i in range(n_qbits)]
if q_i == -2:
	A[0]=1
else:
	A_norm=0
	for k in range(2**n_qbits):
		if k >= 2**q_i -1 and k <= 2**(q_f+1)-1: 
			A[k] = random.uniform(-1,1)+1j*random.uniform(-1,1)
			if q_i == -2 and k == 0: A[k] = 1

			A_norm+=np.absolute(A[k])**2
	A=A/np.sqrt(A_norm)
C=A


printf('Initial state: |psi> = '),
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
		before1, sep1, after1 = before.split()[1].rpartition(":")
		g=before.split( )[0]
	else: g = ''
############ 1-qubit gates
################### gate id

	if g=='id' or g=='h' or g=='x' or g=='y' or g=='z' or g=='s' or g=='sdg' or g=='t' or g=='tdg' or g=='measure':
		qbit_i,qbit_f,q,q = get_qbits(command)
		if g =='id':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = ID(n_qbits,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = ID(n_qbits,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					
################### gate h
		if g=='h':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = H(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	h Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)					
					'''
					for m in range(n_qbits):
						if m != qbit: 
							cmd = "echo  '\t	nop Q%s'  >> QS.qasm"%(m)
							subprocess.call(cmd, shell=True)
				'''
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = H(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
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
					B = X(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	X Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = X(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	X Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate y
		if g =='y':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Y(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Y Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Y(n_qbits,qbit,A,B)	
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Y Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate z					
		if g =='z':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Z(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Z Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Z(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Z Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate s
		if g =='s':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = S(n_qbits,qbit,A,B)	
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	S Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = S(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	S Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate sdg
		if g =='sdg':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Sdg(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Sd Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Sdg(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Sd Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate t
		if g =='t':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = T(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	T Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = T(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	T Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)	

################### gate tdg
		if g =='tdg':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Tdg(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Td Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Tdg(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Td Q%s'  >> QS.qasm"%(qbit)
					subprocess.call(cmd, shell=True)

################### gate sk
	if g =='sk':
		qbit_i,qbit_f,k,k1 = get_qbits(command)
		klog=int(np.log2(k))
		cs='Sk'+str(klog)
		if qbit_f >= qbit_i:
			for qbit in range(qbit_i,qbit_f+1):
				B = Sk(n_qbits,qbit,k,A,B)	
				A,C = print_state(g,n_qbits,verbose,A,B,C)
				cmd = "echo  '\t	%s Q%s'  >> QS.qasm"%(cs,qbit)
				subprocess.call(cmd, shell=True)
		elif qbit_f < qbit_i:
			for qbit in range(qbit_i,qbit_f-1,-1):
				B = Sk(n_qbits,qbit,k,A,B)
				A,C = print_state(g,n_qbits,verbose,A,B,C)
				cmd = "echo  '\t	%s Q%s'  >> QS.qasm"%(cs,qbit)
				subprocess.call(cmd, shell=True)


############ 2-qubit gates
################### gate cx
	if g == 'cx':
		qbit_c_i,qbit_c_f,qbit_t_i,qbit_t_f = get_qbits(command)
		if qbit_c_f > qbit_c_i and qbit_t_i == qbit_t_f: # qbit_c_f > qbit_c_i qbit_t_i == qbit_t_f
			qbit_t = qbit_t_i
			for qbit_c in range(qbit_c_i,qbit_c_f+1):
				B = CX(n_qbits,qbit_c,qbit_t,A,B)
				A,C = print_state(g,n_qbits,verbose,A,B,C)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				subprocess.call(cmd, shell=True)
		elif qbit_c_f < qbit_c_i and qbit_t_i == qbit_t_f: # qbit_c_f < qbit_c_i qbit_t_i == qbit_t_f
			qbit_t = qbit_t_i
			for qbit_c in range(qbit_c_i,qbit_c_f-1,-1):
				B = CX(n_qbits,qbit_c,qbit_t,A,B)
				A,C = print_state(g,n_qbits,verbose,A,B,C)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				subprocess.call(cmd, shell=True)
		elif qbit_c_f == qbit_c_i and qbit_t_f >= qbit_t_i: # qbit_c_f == qbit_c_i and qbit_t_f > qbit_t_i
			qbit_c = qbit_c_i
			for qbit_t in range(qbit_t_i,qbit_t_f+1):
				B = CX(n_qbits,qbit_c,qbit_t,A,B)
				A,C = print_state(g,n_qbits,verbose,A,B,C)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				subprocess.call(cmd, shell=True)
		elif qbit_c_f == qbit_c_i and qbit_t_i >= qbit_t_f: # qbit_c_f == qbit_c_i and qbit_t_i < qbit_t_f
			qbit_c = qbit_c_i
			for qbit_t in range(qbit_t_i,qbit_t_f-1,-1):
				B = CX(n_qbits,qbit_c,qbit_t,A,B)				
				A,C = print_state(g,n_qbits,verbose,A,B,C)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				subprocess.call(cmd, shell=True)

################### gate csk
	if g == 'csk':
		qbit_c,qbit_t,k,k1 = get_qbits(command)
		klog=int(np.log2(k))
		cs='CSk'+str(klog)
		B = CSk(n_qbits,qbit_c,qbit_t,k,A,B)	
		A,C = print_state(g,n_qbits,verbose,A,B,C)
		cmd = "echo  '\t	%s Q%s,Q%s'  >> QS.qasm"%(cs,qbit_c,qbit_t)
		subprocess.call(cmd, shell=True)
		'''for m in range(n_qbits):
			if m != qbit_t: 
				cmd = "echo  '\t	nop Q%s'  >> QS.qasm"%(m)
				subprocess.call(cmd, shell=True)
'''



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

P=[0 for i in range(2**np.sum(M))]
Amp=['' for i in range(2**np.sum(M))]

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
	if np.absolute(C[i]) > 0:
		if len(Amp[num]) == 0:
			Amp[num] = Amp[num] + str('({:.3f}'.format(C[i]/np.sqrt(P[num]))) + ')' + '|' + str(s_i) + '>'
		else:
			Amp[num] = Amp[num] + str(' + ({:.3f}'.format(C[i]/np.sqrt(P[num]))) + ')' + '|' + str(s_i) + '>'



if  np.sum(M) >0:
	print('\nProbabilities after measurement:')
else:
	print('\nFinal state:')

for i in range(2**np.sum(M)):
	s_i = ("{:0%db}"%np.sum(M)).format(i)[::-1]
	if P[i] != 0:
		if  np.sum(M) >0:
			printf('\nP(' + str(s_i) + ') = '),
			print(P[i])
		printf('|psi> = '),
		print(Amp[i])
		

subprocess.call( "python qasm2tex.py QS.qasm > circ.tex",shell=True)
subprocess.call("latex circ.tex >/dev/null 2>&1",shell=True)
subprocess.call("dvips -D2400 -E circ.dvi >/dev/null 2>&1",shell=True)
cmd_exists = lambda x: any(os.access(os.path.join(path, x), os.X_OK) for path in os.environ["PATH"].split(os.pathsep))
if cmd_exists('latex') == True:
	print('\nIf latex is installed correctly then figure circ.ps was created\n')

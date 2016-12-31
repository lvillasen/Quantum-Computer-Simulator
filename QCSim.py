# Quantum Computer Simulator
# Luis Villasenor
# lvillasen@gmail.com
# 12/16/2016 Notation compatible with IBM's Quantum Experience (http://www.research.ibm.com/quantum/)
# 12/18/2016 added compact notation to expand gates applied to sequences of qubits 
# 12/26/2016 added init command to initialize qubits 
# 12/28/2016 added  qasm2tex.py code from I. Chuang to plot circuit (https://www.media.mit.edu/quanta/qasm2circ/)

# Usage
# python QCSim.py samples/example_Bell.txt
from __future__ import print_function
import numpy as np
import sys
import string
import os
import random

def printf(str, *args):
    print(str % args, end='')

def set_bit(value, bit):
    return value | (1<<bit)

def clear_bit(value, bit):
    return value & ~(1<<bit)

def print_state(g,n_qbits,verbose,A,B,C):
	if g != 'cx': print('Gate',g,'on qubit', qbit), 
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
	before1, sep1, after1 = before.split()[1].rpartition(":")
	g=before.split( )[0]
	if g != 'cx':
		if sep1 == ':': 
			a=[int(s) for s in before1.split()[0] if s.isdigit()]
			if len(a)==1:qbit_i= a[0]
			if len(a)==2:qbit_i= 10*a[0]+a[1]
			a=[int(s) for s in after1.split()[0] if s.isdigit()]
			if len(a)==1:qbit_f= a[0]
			if len(a)==2:qbit_f= 10*a[0]+a[1]
		else:
			a=[int(s) for s in before.split()[1] if s.isdigit()]
			if len(a)==1:qbit_i= a[0]
			if len(a)==2:qbit_i= 10*a[0]+a[1]
			qbit_f=qbit_i
		qbit_c_i = qbit_i
		qbit_c_f = qbit_f
		qbit_t_i = -1
		qbit_t_f = -1
	else:
		if sep1 == ':': 
			a=[int(s) for s in before1.split()[0] if s.isdigit()]
			if len(a)==1:qbit_c_i= a[0]
			if len(a)==2:qbit_c_i= 10*a[0]+a[1]
			a=[int(s) for s in after1.split()[0] if s.isdigit()]
			if len(a)==1:qbit_c_f= a[0]
			if len(a)==2:qbit_c_f= 10*a[0]+a[1]
		else:
			a=[int(s) for s in before.split()[1] if s.isdigit()]
			if len(a)==1:qbit_c_i= a[0]
			if len(a)==2:qbit_c_i= 10*a[0]+a[1]
			qbit_c_f=qbit_c_i
		before2, sep2, after2 = before.split()[2].rpartition(":")
		if sep2 == ':': 
			a=[int(s) for s in before2.split()[0] if s.isdigit()]
			if len(a)==1:qbit_t_i= a[0]
			if len(a)==2:qbit_t_i= 10*a[0]+a[1]
			a=[int(s) for s in after2.split()[0] if s.isdigit()]
			if len(a)==1:qbit_t_f= a[0]
			if len(a)==2:qbit_t_f= 10*a[0]+a[1]
		else:
			a=[int(s) for s in before.split()[2] if s.isdigit()]
			if len(a)==1:qbit= a[0]
			if len(a)==2:qbit= 10*a[0]+a[1]
			qbit_t_i=qbit
			qbit_t_f=qbit
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
	before1, sep1, after1 = before.split()[1].rpartition(":")
	g=before.split( )[0]
	if g == 'init' or g =='id' or g=='h' or g=='x' or g=='y' or g=='z' or g=='s' or g=='sdg' or g=='t' or g=='tdg' or g=='measure' or g == 'verbose':
		qbit_i,qbit_f,q,q = get_qbits(command)
		n_qbits=max(n_qbits, qbit_i+1)
		n_qbits=max(n_qbits, qbit_f+1)
		if g == 'init': 
			q_i=qbit_i
			q_f=qbit_f	
		if g == 'verbose':
			verbose = qbit_i	
	elif g == 'cx':
		qbit_c_i,qbit_c_f,qbit_t_i,qbit_t_f = get_qbits(command)
		n_qbits=max(n_qbits, qbit_c_f+1)
		n_qbits=max(n_qbits, qbit_t_f+1)

print('\nNumber of qubits: ',n_qbits)
os.system("echo '# Quantum score'  > QS.qasm")
Sd="\'"+str('{S}^{\\dagger}')+"\'"
cmd = 'echo "\t 	def	Sd,0,%s"  >> QS.qasm'%(Sd)
os.system(cmd)
Td="\'"+str('{T}^{\\dagger}')+"\'"
cmd = 'echo "\t 	def	Td,0,%s"  >> QS.qasm'%(Td)
os.system(cmd)
for qbit in range(n_qbits):
	if qbit >= q_i and qbit <= q_f:
		cmd = 'echo "\t	qubit Q%s"  >> QS.qasm'%(qbit)
		os.system(cmd)
	else:
		cmd = 'echo "\t	qubit Q%s,0"  >> QS.qasm'%(qbit)
		os.system(cmd)


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
	before1, sep1, after1 = before.split()[1].rpartition(":")
############ 1-qubit gates
################### gate id
	g = before.split( )[0]
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
					os.system(cmd)		
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = H(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	h Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	

################### gate x			
		if g=='x':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = X(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	X Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = X(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	X Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	

################### gate y
		if g =='y':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Y(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Y Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Y(n_qbits,qbit,A,B)	
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Y Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	

################### gate z					
		if g =='z':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Z(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Z Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Z(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Z Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	

################### gate s
		if g =='s':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = S(n_qbits,qbit,A,B)	
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	S Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = S(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	S Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	

################### gate sdg
		if g =='sdg':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Sdg(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Sd Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Sdg(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Sd Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	

################### gate t
		if g =='t':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = T(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	T Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = T(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	T Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)	

################### gate tdg
		if g =='tdg':
			if qbit_f >= qbit_i:
				for qbit in range(qbit_i,qbit_f+1):
					B = Tdg(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Td Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)
			elif qbit_f < qbit_i:
				for qbit in range(qbit_i,qbit_f-1,-1):
					B = Tdg(n_qbits,qbit,A,B)
					A,C = print_state(g,n_qbits,verbose,A,B,C)
					cmd = "echo  '\t	Td Q%s'  >> QS.qasm"%(qbit)
					os.system(cmd)

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
				os.system(cmd)
		elif qbit_c_f < qbit_c_i and qbit_t_i == qbit_t_f: # qbit_c_f < qbit_c_i qbit_t_i == qbit_t_f
			qbit_t = qbit_t_i
			for qbit_c in range(qbit_c_i,qbit_c_f-1,-1):
				B = CX(n_qbits,qbit_c,qbit_t,A,B)
				A,C = print_state(g,n_qbits,verbose,A,B,C)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				os.system(cmd)
		elif qbit_c_f == qbit_c_i and qbit_t_f >= qbit_t_i: # qbit_c_f == qbit_c_i and qbit_t_f > qbit_t_i
			qbit_c = qbit_c_i
			for qbit_t in range(qbit_t_i,qbit_t_f+1):
				B = CX(n_qbits,qbit_c,qbit_t,A,B)
				A,C = print_state(g,n_qbits,verbose,A,B,C)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				os.system(cmd)
		elif qbit_c_f == qbit_c_i and qbit_t_i >= qbit_t_f: # qbit_c_f == qbit_c_i and qbit_t_i < qbit_t_f
			qbit_c = qbit_c_i
			for qbit_t in range(qbit_t_i,qbit_t_f-1,-1):
				B = CX(n_qbits,qbit_c,qbit_t,A,B)				
				A,C = print_state(g,n_qbits,verbose,A,B,C)
				cmd = "echo  '\t	cnot Q%s,Q%s'  >> QS.qasm"%(qbit_c,qbit_t)
				os.system(cmd)

################### measure
	if g == 'measure':
		if qbit_f >= qbit_i:
			for qbit in range(qbit_i,qbit_f+1):
				M[qbit] = 1
				print('Measure qubit', qbit)
				cmd = "echo  '\t 	measure Q%s'  >> QS.qasm"%(qbit)
				os.system(cmd)	 
		elif qbit_f < qbit_i:
			for qbit in range(qbit_i,qbit_f-1,-1):
				M[qbit] = 1
				print('Measure qubit', qbit)
				cmd = "echo  '\t	measure Q%s'  >> QS.qasm"%(qbit)
				os.system(cmd)	 

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



print('\nProbabilities after measurement:')
for i in range(2**np.sum(M)):
	s_i = ("{:0%db}"%np.sum(M)).format(i)[::-1]
	if P[i] != 0: 
		printf('\nP(|' + str(s_i) + '>) = '),
		print(P[i])
		printf('|psi> = '),
		print(Amp[i])
		

os.system("python qasm2tex.py QS.qasm > circ.tex")
os.system("latex circ.tex >/dev/null 2>&1")
os.system("dvips -D2400 -E circ.dvi >/dev/null 2>&1")
cmd_exists = lambda x: any(os.access(os.path.join(path, x), os.X_OK) for path in os.environ["PATH"].split(os.pathsep))
if cmd_exists('latex') == True:
	print('\nIf latex is installed correctly then figure circ.ps was created\n')

# Quantum Computer Simulator
# Luis Villasenor
# lvillasen@gmail.com
# 12/16/2016
# Usage
# python QCSim.py prog.ql
from __future__ import print_function
import numpy as np
import sys
import string
def printf(str, *args):
    print(str % args, end='')
def set_bit(value, bit):
    return value | (1<<bit)
def clear_bit(value, bit):
    return value & ~(1<<bit)
if len(sys.argv) > 1:
	file=sys.argv[1]
f = open(file,"r") #opens file with qc program
myList = []
for line in f:
    myList.append(line)
n_qbits =0
for i in range(len(myList)):
	before, sep, after = myList[i].rpartition(";")
	g=before.split( )[0]
	if g=='id' or g=='h' or g=='x' or g=='y' or g=='z' or g=='s' or g=='sdg' or g=='t' or g=='tdg':
		a=[int(s) for s in before.split()[1] if s.isdigit()]
		if len(a)==1:qbit= a[0]
		if len(a)==2:qbit= 10*a[0]+a[1]
		n_qbits=max(n_qbits, qbit+1)
	if g=='cx':
		a=[int(s) for s in before.split()[1] if s.isdigit()]
		qbit_c= a[0]
		a=[int(s) for s in before.split()[2] if s.isdigit()]
		qbit_t= a[0]
		n_qbits=max(n_qbits, qbit_c+1)
		n_qbits=max(n_qbits, qbit_t+1)
print('Number of qbits: ',n_qbits)
A = [0 for i in range(2**n_qbits)]
B = [0 for i in range(2**n_qbits)]
C = [0 for i in range(2**n_qbits)]
M = [0 for i in range(n_qbits)]
m=0
A[0]=1
printf('Initial state: |psi> = '),
for i in range(2**n_qbits):
	s_i=("{:0%db}"%n_qbits).format(i)[::-1]
	if A[i] != 0: 
		printf(str(A[i])), 
		printf(str('|'+s_i+'>')),
printf('\n')
for i in range(len(myList)):
	before, sep, after = myList[i].rpartition(";")
	p=0
############ 1-qbit gates
	g=before.split( )[0]
	if g=='id' or g=='h' or g=='x' or g=='y' or g=='z' or g=='s' or g=='sdg' or g=='t' or g=='tdg':
		a=[int(s) for s in before.split()[1] if s.isdigit()]
		if len(a)==1:qbit= a[0]
		if len(a)==2:qbit= 10*a[0]+a[1]
		if g =='id':
			for j in range(2**n_qbits):
				if A[j] != 0:
					B[j]+=A[j]
		if g=='h':
			for j in range(2**n_qbits):
				if A[j] != 0:
					bit_parity=(j>>qbit)%2
					if bit_parity ==0:
						B[j]+=1/np.sqrt(2)*A[j]
						B[set_bit(j,qbit)]+=1/np.sqrt(2)*A[j]
					if bit_parity ==1:
						B[clear_bit(j,qbit)]+=1/np.sqrt(2)*A[j]
						B[j]+=-1/np.sqrt(2)*A[j]
		if g=='x':
			for j in range(2**n_qbits):
				if A[j] != 0:
					bit_parity=(j>>qbit)%2
					if bit_parity ==0:
						B[set_bit(j,qbit)]+=A[j]
					if bit_parity ==1:
						B[clear_bit(j,qbit)]+=A[j]	
		if g =='y':
			for j in range(2**n_qbits):
				if A[j] != 0:
					bit_parity=(j>>qbit)%2
					if bit_parity ==0:
						B[set_bit(j,qbit)]+=1j*A[j]
					if bit_parity ==1:
						B[clear_bit(j,qbit)]+=-1j*A[j]	
		if g =='z':
			for j in range(2**n_qbits):
				if A[j] != 0:
					bit_parity=(j>>qbit)%2
					if bit_parity ==0:
						B[j]+=A[j]
					if bit_parity ==1:
						B[j]+=-A[j]	
		if g =='s':
			for j in range(2**n_qbits):
				if A[j] != 0:
					bit_parity=(j>>qbit)%2
					if bit_parity ==0:
						B[j]+=A[j]
					if bit_parity ==1:
						B[j]+=1j*A[j]	
		if g =='sdg':
			for j in range(2**n_qbits):
				if A[j] != 0:
					bit_parity=(j>>qbit)%2
					if bit_parity ==0:
						B[j]+=A[j]
					if bit_parity ==1:
						B[j]+=-1j*A[j]
		if g =='t':
			for j in range(2**n_qbits):
				if A[j] != 0:
					bit_parity=(j>>qbit)%2
					if bit_parity ==0:
						B[j]+=A[j]
					if bit_parity ==1:
						B[j]+=1/np.sqrt(2)*(1+1j)*A[j]
		if g =='tdg':
			for j in range(2**n_qbits):
				if A[j] != 0:
					bit_parity=(j>>qbit)%2
					if bit_parity ==0:
						B[j]+=A[j]
					if bit_parity ==1:
						B[j]+=1/np.sqrt(2)*(1-1j)*A[j]
		print('Gate',g,'on qbit', qbit), 
		p=1
############ 2-qbit gates
	if g=='cx':
		a=[int(s) for s in before.split()[1] if s.isdigit()]
		if len(a)==1:qbit_c= a[0]
		if len(a)==2:qbit_c= 10*a[0]+a[1]
		a=[int(s) for s in before.split()[2] if s.isdigit()]
		if len(a)==1:qbit_t= a[0]
		if len(a)==2:qbit_t= 10*a[0]+a[1]
		for j in range(2**n_qbits):
			if A[j] != 0:
				bit_parity_c=(j>>qbit_c)%2
				bit_parity_t=(j>>qbit_t)%2
				if bit_parity_c ==0:
					B[j]+=A[j]
				else:
					if bit_parity_t==0:
						B[set_bit(j,qbit_t)]+=A[j]
					else:
						B[clear_bit(j,qbit_t)]+=A[j]
		print('Gate cx on control qbit =', qbit_c,' and target qbit =',qbit_t),
		p=1
	if p==1:
		printf('\tresulted in state |psi> = '),
		k1=0
		psi=''
		for k in range(2**n_qbits):
			s_i=("{:0%db}"%n_qbits).format(k)[::-1]
			if B[k] != 0: 
				k1+=1
				if k1==1:psi+=str(B[k])+'|'+s_i+'> '
				else:psi+='+ '+str(B[k])+str('|'+s_i+'> ')
		psi=string.replace(psi,'+ -', '- ')
		print(psi)
		print
		for  k in range(2**n_qbits): C[k]=B[k]
	for  k in range(2**n_qbits): A[k]=B[k]
	B = [0 for i in range(2**n_qbits)]
	if g =='measure':
		a=[int(s) for s in before.split()[1] if s.isdigit()]
		qbit= a[0]
		M[qbit]=1
		m+=2**qbit
		print('Measure qbit', qbit) 
P=[0 for i in range(2**np.sum(M))]
for i in range(2**n_qbits):
	s_i=("{:0%db}"%n_qbits).format(i)
	num=0
	k=0
	for j in range(len(M)):
		if M[j]==1:
			num+=((i>>j)&1)*2**k
			k+=1
	P[num]+=+np.absolute(C[i])**2
print('Probabilities after measurement:')
for i in range(2**np.sum(M)):
	s_i=("{:0%db}"%np.sum(M)).format(i)[::-1]
	if P[i]!=0: 
		printf('P('+str(s_i)+') = '),
		print(P[i])	


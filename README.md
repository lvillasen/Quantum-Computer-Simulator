# Quantum Computer Simulator
 Luis Villasenor
 
 lvillasen@gmail.com
 
Date: 12-15-2016

#Ideal simulator of a quantum computer 
Written in Python

Compatible with IBM Quantum Experience

#Usage 

python QCSim.py prog.ql

where:

prog.ql is the program to be simulated

The syntax is fully compatible with that used in the IBM Quantum 

Experience webpage (https://quantumexperience.ng.bluemix.net/)

The number of qubits is limited in a natural way by the resources 

available on the system used to run this program

As usual, qubits are ordered from left to right on the quantum states psi>

#For instance, python QCSim.py prog.ql produces:

Number of qbits:  3

Initial state: |psi> = 1|000>

Gate h on qbit 0

	resulted in state |psi> = 0.707106781187|000> + 0.707106781187|100>
	
Gate h on qbit 1

	resulted in state |psi> = 0.5|000> + 0.5|100> + 0.5|010> + 0.5|110>
	
Gate x on qbit 2

	resulted in state |psi> = 0.5|001> + 0.5|101> + 0.5|011> + 0.5|111> 
	
Gate cx on control qbit = 1  and target qbit = 2

	resulted in state |psi> = 0.5|010> + 0.5|110> + 0.5|001> + 0.5|101> 
	
Gate cx on control qbit = 0  and target qbit = 2

	resulted in state |psi> = 0.5|100> + 0.5|010> + 0.5|001> + 0.5|111> 
	
Gate h on qbit 0

	resulted in state |psi> = 0.353553390593|000> - 0.353553390593|100> +
	
0.353553390593|010> + 0.353553390593|110> + 0.353553390593|001> +

0.353553390593|101> + 0.353553390593|011> - 0.353553390593|111>

Gate h on qbit 1

	resulted in state |psi> = 0.5|000> - 0.5|110> + 0.5|001> + 0.5|111> 
	
Gate h on qbit 2

	resulted in state |psi> = 0.707106781187|000> - 0.707106781187|111> 
	
Measure qbit 0

Measure qbit 1

Measure qbit 2

Probabilities after measurement:

P(000) = 0.5

P(111) = 0.5

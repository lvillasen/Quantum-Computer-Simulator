# Quantum Computer Simulator
 Luis Villasenor
 
 lvillasen@gmail.com
 
#Simulator of a quantum computer written in Python

Fully compatible with the syntax and output code of IBM's Quantum Experience

	IBM Quantum Experience (https://quantumexperience.ng.bluemix.net/)

#Usage 

	python QCSim.py prog.ql

where:

prog.ql is one example that illustrates the syntax of this particular quantum simulation language.  

For many more examples, the possibility to use a real quantum computer and 

an excellent tutorial, you can visit the website "IBM Quantum Experience"

This is an ideal simulator. The maximum number of qubits it can handle is limited in a natural way 

by the resources available on the system used to run these programs

It implements an automatic extension of the range of qubits in commands, for instance
	
	h q[0:1];
	
is equivalent to

	h q[0];
	
	h q[1];

As usual, qubits are ordered from left to right on the quantum states |psi>

For instance, to simulate the example provided, you just type

	python QCSim.py prog.ql 

It produces the following output:

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

The same program can be written as

	h q[0:1];
	
	x q[2];
	
	cx q[1:0], q[2];
	
	h q[0:2];
	
	measure q[0:2];

# Quantum Computer Simulator written in Python
 Luis Villasenor
 
 lvillasen@gmail.com

Compatible with the syntax and output code of IBM's Quantum Experience

	IBM Quantum Experience (https://quantumexperience.ng.bluemix.net/)

#Usage 

	python QCSim.py samples/example_GHZ.txt 

where:

example_GHZ.txt is one example that illustrates the syntax of this particular quantum simulation language.  

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

The output of

	python QCSim.py samples/example_GHZ.txt

is

	Number of qubits:  3
	Initial state: |psi> = (1.000)|000> 
	Gate h on qubit 0
	Gate h on qubit 1
	Gate x on qubit 2
	Gate cx on control qubit 1  and target qubit 2
	Gate cx on control qubit 0  and target qubit 2
	Gate h on qubit 0
	Gate h on qubit 1
	Gate h on qubit 2
	Measure qubit 0

	Probabilities after measurement:

	P(|0>) = 0.5
	|psi> = (1.000)|000>

	P(|1>) = 0.5
	|psi> = (-1.000)|111>

	If latex is installed correctly then figure circ.ps was created

The example

	h q[0];
	h q[1];
	x q[2];
	cx q[1], q[2];
	cx q[0], q[2];
	h q[0];
	h q[1];
	h q[2];
	measure q[0];
	measure q[1];
	measure q[2];

can also be written as

	h q[0:1];
	x q[2];
	cx q[1:0], q[2];
	h q[0:2];
	measure q[0:2];

The command

	init q[k];
	
initializes the first k qubits to a random superposition
	
If the latex command is found, the circuit is created in ps format by using the qasm2tex.py code from I. Chuang (https://www.media.mit.edu/quanta/qasm2circ/)

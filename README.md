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

#Brief description

This is an ideal simulator. The maximum number of qubits it can handle is limited in a natural way 

by the resources available on the system used to run these programs. For instance, 20-qubit programs run

in a few seconds on a typical laptop

The commands implemented are the following:

*	init q[i]; initializes qubit i to a random superposition

*	id q[i]; identity gate applied to qubit i

*	h q[i]; Hadamard gate H applied to qubit i

*	x q[i]; X Pauli gate applied to qubit i

*	y q[i]; Y Pauli gate applied to qubit i

*	z q[i]; Z Pauli gate applied to qubit i

*	s q[i]; S phase shift applied to qubit i with a phase shift of pi

*	sdg q[i]; Hermitian conjugate of the S phase shift applied to qubit i

*	t q[i]; T phase shift applied to qubit i with a phase shift of pi/2

*	tdg q[i]; Hermitian conjugate of the T phase shift applied to qubit i

*	sk q[i]; S(pi/k) phase shift applied to qubit i with a phase shift of pi/k where k is any integer

*	cx q[i], q[j]; CNOT gate applied to control qubit i and target qubit j

*	csk q[i], q[j], k; controlled S(pi/k) phase shift applied to control qubit i and target qubit j with k integer

*	measure q[i]; measure state of qubit i in the standard basis

*	verbose 0(1); verbose mode off(on)


Comments can be merged with code after the ; terminator or in new lines

A bash script is provided to automatically generate quantum Fourier transform (QFT) programs for any number of qubits

The Python code implements an automatic extension of the range of qubits in commands, for instance
	
	h q[0:4];
	
is equivalent to

	h q[0];
	h q[1];
	h q[2];
	h q[3];
	h q[4];

and 

	cx q[3:1], q[4];

is equivalent to

	cx q[3], q[4];
	cx q[2], q[4];
	cx q[1], q[4];


The initial state is

	|000...00>

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

	P(0) = 0.5
	|psi> = (1.000)|000>
	
	P(1) = 0.5
	|psi> = (-1.000)|111>

	If latex is installed correctly then figure circ.ps was created

If we add verbose mode

	verbose 1;
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
	
the output becomes

	Number of qubits:  3
	Initial state: |psi> = (1.000)|000> 
	Gate h on qubit 0
	  resulted in state |psi> = (0.707)|000> + (0.707)|100> 
	Gate h on qubit 1
	  resulted in state |psi> = (0.500)|000> + (0.500)|100> + (0.500)|010> + (0.500)|110> 
	Gate x on qubit 2
	  resulted in state |psi> = (0.500)|001> + (0.500)|101> + (0.500)|011> + (0.500)|111> 
	Gate cx on control qubit 1  and target qubit 2
	  resulted in state |psi> = (0.500)|010> + (0.500)|110> + (0.500)|001> + (0.500)|101> 
	Gate cx on control qubit 0  and target qubit 2
	  resulted in state |psi> = (0.500)|100> + (0.500)|010> + (0.500)|001> + (0.500)|111> 
	Gate h on qubit 0
	  resulted in state |psi> = (0.354)|000> + (-0.354)|100> + (0.354)|010> + (0.354)|110> + (0.354)|001> + (0.354)|101> + 		(0.354)|011> + (-0.354)|111> 
	Gate h on qubit 1
	  resulted in state |psi> = (0.500)|000> + (-0.500)|110> + (0.500)|001> + (0.500)|111> 
	Gate h on qubit 2
	  resulted in state |psi> = (0.707)|000> + (-0.707)|111> 
	Measure qubit 0

	Probabilities after measurement:

	P(0) = 0.5
	|psi> = (1.000)|000>

	P(1) = 0.5
	|psi> = (-1.000)|111>

	If latex is installed correctly then figure circ.ps was created

This example can also be written as

	verbose 1;
	h q[0:1];
	x q[2];
	cx q[1:0], q[2];
	h q[0:2];
	measure q[0:2];
	
If the latex command is found, the circuit is created in ps format by using the qasm2tex.py code from I. Chuang (https://www.media.mit.edu/quanta/qasm2circ/)

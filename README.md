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

	idn q[i]; identity gate applied to qubit i

	h q[i]; Hadamard gate H applied to qubit i

	x q[i]; X Pauli gate applied to qubit i

	y q[i]; Y Pauli gate applied to qubit i

	z q[i]; Z Pauli gate applied to qubit i

	s q[i]; S phase shift applied to qubit i with a phase shift of \pi

	sdag q[i]; Hermitian conjugate of the S phase shift applied to qubit i

	t q[i]; T phase shift applied to qubit i with a phase shift of $\pi$/2

	tdag q[i]; Hermitian conjugate of the T phase shift applied to qubit i

	sk q[i]; S(\pi/k) phase shift applied to qubit i with a phase shift of \pi/k where k is any integer

	cx q[i], q[j]; CNOT gate applied to control qubit i and target qubit j

	csk q[i], q[j], k; controlled S(\pi/k) phase shift applied to control qubit i and target qubit j with k integer

	measure q[i]; measure state of qubit i in the standard basis

	verbose 0(1); verbose mode off(on)


The Python code implements an automatic extension of the range of qubits in commands, for instance
	
	h q[0:4];
	
is equivalent to

	h q[0];
	h q[1];
	h q[2];
	h q[3];

and 

	cx q[1:3], q[4];

is equivalent to

	cx q[1], q[4];

	cx q[2], q[4];

	cx q[3], q[4];


The initial state is

	|000...00>

As usual, qubits are ordered from left to right on the quantum states |psi>

The output of

	python QCSim.py samples/example_GHZ.txt

is

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
	
	If latex is installed figure circ.ps was created

This example

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

can also be written as

	verbose 1;
	h q[0:1];
	x q[2];
	cx q[1:0], q[2];
	h q[0:2];
	measure q[0:2];
	
If the latex command is found, the circuit is created in ps format by using the qasm2tex.py code from I. Chuang (https://www.media.mit.edu/quanta/qasm2circ/)

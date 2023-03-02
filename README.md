# Quantum Computer Simulator written in Python
 Luis Villasenor
 
 lvillasen@gmail.com

Compatible with the syntax and output code of IBM's Quantum Experience

	IBM Quantum Experience (https://quantumexperience.ng.bluemix.net/)

# Usage 

	python QCSim.py samples/example_YYY.txt 

where:

example_YYY.txt is one of the examples provided.  

For many more examples, the possibility to use a real quantum computer and 

an excellent tutorial, you can visit the website "IBM Quantum Experience"

# Brief description

This is an ideal simulator. The unitary operations are optimized with matrix-free algorithms. 

The maximum number of qubits it can handle is limited, in a natural way, by the resources available on the system used to run the programs. For instance, the 18-qubit Grover-search example provided (402 repetition of the basic search algorithm) runs in a few seconds on a typical laptop.

The gates and commands implemented so far are the following:

*	**init q[i];** initializes qubit i to a random superposition

*	**id q[i];** identity gate applied to qubit i

*	**h q[i];** Hadamard gate H applied to qubit i

*	**x q[i];** X Pauli gate applied to qubit i

*	**y q[i];** Y Pauli gate applied to qubit i

*	**z q[i];** Z Pauli gate applied to qubit i

*	**s q[i];** S phase shift applied to qubit i with a phase shift of pi

*	**sdg q[i];** Hermitian conjugate of the S phase shift applied to qubit i

*	**t q[i];** T phase shift applied to qubit i with a phase shift of pi/2

*	**tdg q[i];** Hermitian conjugate of the T phase shift applied to qubit i

*	**QFT q[i:j];** Quantum Fourier transform applied from qbit i to qbit j  

*	**IQFT q[i:j];** Inverse quantum Fourier transform applied from qbit i to qbit j  

*	**sk q[i], k;** S(pi/k) phase shift applied to qubit i with a phase shift of pi/k where k is any integer

*	**cx q[i], q[j];** CNOT gate applied to control qubit i and target qubit j

*	**csk q[i], q[j], k;** controlled S(pi/k) phase shift applied to control qubit i and target qubit j with k integer

*	**measure q[i];** measure state of qubit i in the standard basis

*	**verbose 0(1);** verbose mode off(on)

*	**Sign i;** flips sign of states with index i in the standard basis

*	**plot 0(1);** plot off(on), default 0

*	**printout 0(1);** plot off(on), default 0

*	**Inverse_P_threshold i;** if printout is set to 1, only the basis states that have probabilities greater than 1/i are printed. This is handy to avoid large printouts when the number of qubits is large

Lines that do not terminate with a semicolon are treated as comments,
they can be merged with code after the ; terminator or in new lines

In addition to the QFT and IQFT commands, a bash script (QFT.sh) is provided to automatically generate quantum Fourier transform (QFT) and inverse Fourier transform (IQFT) programs based on elementary gates for any number of qubits.

A bash script (Grover.sh) is also provided to automatically generate Grover search programs for any number of qubits

The code implements an automatic extension of the range of qubits in commands, for instance
	
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

If "printout" is set to 1, the basis states with probabilities > 1/Inverse_P_threshold  are printed at the end 

If "plot" is set to 1, the probabilities of all the basis states are plotted

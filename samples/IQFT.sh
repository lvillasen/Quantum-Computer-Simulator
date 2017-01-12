#!/bin/bash
# Bash script to generate IQFT examples for any number of qubits
#Autor: Luis Villaseñor
#Date: 1/10/2017 
# Usage: bash samples/IQFT.sh > samples/example_IQFT.txt
# Usage: python QCSim.py samples/example_IQFT.txt
N=4
echo "Initial state = |0000..>";echo
echo 'IQFT code starts here'

for i in $(seq $((N-1)) 1); do
	echo "h q[$i];";
	for j in $(seq $((i-1)) 0); do
		echo "csk q[$i], q[$j], $(( 2 ** $((i-j)) ));"
	done
	echo
	
done
echo "h q[0];";echo

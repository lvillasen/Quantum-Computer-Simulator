#!/bin/bash
# Bash script to generate IQFT examples for any number of qubits
#Autor: Luis Villaseñor
#Date: 1/10/2017 
# Usage: bash samples/IQFT.sh > samples/example_IQFT.txt
# Usage: python QCSim.py samples/example_IQFT.txt
n=15
echo "IQFT code starts here"

for i in $(seq $((n-1)) 1); do
	echo "h q[$i];";
	for j in $(seq $((i-1)) 0); do
		echo "csk q[$i], q[$j], -$(( 2 ** $((i-j)) ));"
	done
	echo
	
done
echo "h q[0];"
echo "reverse;"
echo "IQFT code ends here"

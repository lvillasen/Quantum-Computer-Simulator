#!/bin/bash
# Bash script to generate QFT examples for any number of qubits
#Autor: Luis Villaseñor
#Date: 1/1/2017 
# Usage: bash samples/QFT.sh > samples/example_IQFT.txt
# Usage: python QCSim.py samples/example_QFT.txt
N=8
echo "h q[0:$((N-1))];";echo
echo "h q[0];";echo
for i in $(seq 1 $((N-1))); do
	for j in $(seq 0 $((i-1))); do
		echo "csk q[$i], q[$j], $(( 2 ** $((i-j)) ));"
	done
	echo "h q[$i];";echo;
done

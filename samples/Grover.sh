#!/bin/bash
# Bash script to generate a Grover search example for any number of qubits
#Autor: Luis Villaseñor
#Date: 1/10/2017 
# Usage: bash samples/Grover.sh > samples/example_Grover.txt
# Usage: python QCSim.py samples/example_Grover.txt
n=5 # Number of qubits
search_index=21 # select index to search
N_states=$((2 ** $n))
i=$(echo "((3.1416*sqrt($N_states))/4+.5)" | bc -l)
N_times=${i%.*}
echo "circuit 0 ;";
echo "Number of qubits = $((n))";
echo "Number of states = $((N_states))";
echo "Search index = $((search_index))";
echo "Number of optimal Grover iterations = $((N_times))";echo
echo "Initial state = |00..00>";echo
echo "h q[0:$((n-1))];";
echo "State is transformed to fully uniform";echo

for i in $(seq 1 $((N_times))); do
	echo "Iteration no: $((i))"
echo "Sign $((search_index));"
echo "h q[0:$((n-1))];";
#echo "Sign 1:$N_states;"
echo "Sign 0;"
echo "h q[0:$((n-1))];";
done
echo "measure 0:$((n-1)) ;"
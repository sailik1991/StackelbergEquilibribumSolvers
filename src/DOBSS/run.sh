#!/bin/bash

# Please note that you need gurobi (http://www.gurobi.com/index) to be installed for this to work

# Use the one your heart desires. Comment out the rest.
gurobi.sh BSG_miqp.py input.txt > output.txt
#gurobi.sh BSG_milp.py input.txt > output.txt
#gurobi.sh BSG_vs_UR.py input.txt >output.txt

# Print output to terminal
cat output.txt

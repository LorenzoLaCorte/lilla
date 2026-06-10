#!/bin/bash
# -----------------------------------------
# Parameter Set Testing
# -----------------------------------------
# This section of the script is responsible for testing different parameter sets
#   either bruteforcing or optimizing over search spaces. 
#
# Follow these steps to run the script:
# 1. Ensure that all necessary dependencies are set before running
#           that the Python alias matches your environment.
# 2. Define the parameter sets and test types to be tested 
#
# -----------------------------------------

# Env 
PY_ALIAS="python3 -m"
SCRIPT="src.optimization.gp_asymmetric"
GENERAL_RESULT_DIR="./results_asymmetric"

# ------------------------------------------------------------------
# Homogeneous Chains:
# - t_coh: global time of coherence
# - p_gen: global generation success probability
# - p_swap: swap success probability
# - w0: global value for the quality of the links
# - nodes: number of nodes in the chain
# - max_dists: maximum distillations applied to any link at any level
# ------------------------------------------------------------------
#   (t_coh,       p_gen,      p_swap, w0,      nodes, max_dists, test_type, seed)
#
HOMOGENEOUS_SETS=(
    # # ------------------------------------------------------------------ distribution at a given distance (Sec.V.A.)
    # #                                                                    bruteforce simulations (Table 2)
    # "360000       0.000000096 0.85    0.36     2      2          bf -1"     # A, Beta=2
    # "720000       0.000015    0.85    0.9      3      2          bf -1"     # B, Beta=2
    # "1400000      0.00092     0.85    0.952    5      1          bf -1"     # C, Beta=1
    # #                                                                    bayesian opt. simulations (Table 2)
    # "1400000      0.00092     0.85    0.952    5      1          gp 42"     # C, Beta=1 - verified
    # "1400000      0.00092     0.85    0.952    5      2          gp 42"     # C, Beta=2 - verified
    "3600000      0.0026      0.85    0.958    11     2          gp 40"     # D, Beta=2 - seeds 42,41 does not work

    # # ------------------------------------------------------------------ impact of distillation (Sec.V.B.)
    # "1400000      0.00092     0.85    0.95     5      0          bf -1"     # C, Beta=0
    # "1400000      0.00092     0.85    0.96     5      0          bf -1"     # C, Beta=0
    # "1400000      0.00092     0.85    0.97     5      0          bf -1"     # C, Beta=0
    # "1400000      0.00092     0.85    0.98     5      0          bf -1"     # C, Beta=0
    # "1400000      0.00092     0.85    0.95     5      3          gp 42"     # C, Beta=3 - verified
    "1400000      0.00092     0.85    0.96     5      3          gp 40"     # C, Beta=3 - seeds 42,41 does not work
    # "1400000      0.00092     0.85    0.97     5      3          gp 42"     # C, Beta=3 - verified
    # "1400000      0.00092     0.85    0.98     5      3          gp 42"     # C, Beta=3 - verified
)


for PARAMETERS in "${HOMOGENEOUS_SETS[@]}"; do
    IFS=' ' read -r -a PARAM_ARRAY <<< "$PARAMETERS"
    
    T_COH="${PARAM_ARRAY[0]}"
    P_GEN="${PARAM_ARRAY[1]}"
    P_SWAP="${PARAM_ARRAY[2]}"
    W0="${PARAM_ARRAY[3]}"
    NODES="${PARAM_ARRAY[4]}"
    MAX_DISTS="${PARAM_ARRAY[5]}"
    TEST_TYPE="${PARAM_ARRAY[6]}"
    SEED="${PARAM_ARRAY[7]}"

    FILENAME="output.txt"
    TMPFILE=$(mktemp)
        
    echo "Running $TEST_TYPE..."

    # Run the Python script with the specified parameters and append the output to TMPFILE
    { time $PY_ALIAS $SCRIPT \
        --nodes="$NODES" \
        --max_dists="$MAX_DISTS" \
        --optimizer="$TEST_TYPE" \
        --filename="$FILENAME" \
        --t_coh="$T_COH" \
        --p_gen="$P_GEN" \
        --p_swap="$P_SWAP" \
        --w0="$W0" \
        --seed="$SEED" \
    ; } 2>&1 | tee -a "$TMPFILE"

    # Extract the time taken and append it to the output file
    echo "Time taken:" >> "$FILENAME"
    tail -n 3 "$TMPFILE" >> "$FILENAME"
    rm "$TMPFILE"

    # Create a folder for the results if it doesn't exist
    RESULT_DIR="$GENERAL_RESULT_DIR/results_${TEST_TYPE}_tcoh${T_COH}_pgen${P_GEN}_pswap${P_SWAP}_w0${W0}_nodes${NODES}_maxdists${MAX_DISTS}_$SEED"
    mkdir -p "$RESULT_DIR"

    # Move the output file and the plots to the results folder
    mv "$FILENAME" "$RESULT_DIR/"
    if ls *${TEST_TYPE}.pdf 1> /dev/null 2>&1; then
        mv *${TEST_TYPE}.pdf "$RESULT_DIR/"
    else
        echo "No plots yielded for optimizer $TEST_TYPE"
    fi
done

# ------------------------------------------------------------------------------------------------------------------------------------
# Heterogeneous Chains:
# - t_coh: list of node-specific time of coherence
# - p_gen: list of link-specific generation probabilities
# - p_swap: swap probability
# - w0: list of qualities of the links
# - nodes: number of nodes in the chain
# - max_dists: maximum distillations applied to any link at any level
# ------------------------------------------------------------------------------------------------------------------------------------
#   (t_coh,                        p_gen,                  p_swap,   w0,                nodes, max_dists, test_type)
# ------------------------------------------------------------------------------------------------------------------------------------
HETEROGENEOUS_SETS=(
    # # ------------------------------------------------------------------ heterogeneous chain (Sec.V.C.)
    # "[1080000,950000,720000,560000]  [0.002588,0.0009187,0.0009082]  0.85      [0.9577,0.9524,0.9523]   4      2          bf -1"
    # # ------------------------------------------------------------------ faulty chains (Sec.V.D.)
    # "[100000,100000,100000,10000]  [0.0025,0.0025,0.0025]  0.85      [0.95,0.95,0.95]   4      2          bf -1"
    # "[100000,100000,100000,100000] [0.0025,0.0025,0.00025] 0.85      [0.95,0.95,0.95]   4      2          bf -1"
    # "[100000,100000,100000,100000] [0.0025,0.0025,0.0025]  0.85      [0.95,0.95,0.90]   4      2          bf -1"
)

for PARAMETERS in "${HETEROGENEOUS_SETS[@]}"; do
    IFS=' ' read -r -a PARAM_ARRAY <<< "$PARAMETERS"
    
    T_COH=$(echo "${PARAM_ARRAY[0]}" | sed 's/\[//g' | sed 's/\]//g' | tr ',' ' ')
    P_GEN=$(echo "${PARAM_ARRAY[1]}" | sed 's/\[//g' | sed 's/\]//g' | tr ',' ' ')
    P_SWAP="${PARAM_ARRAY[2]}"
    W0=$(echo "${PARAM_ARRAY[3]}" | sed 's/\[//g' | sed 's/\]//g' | tr ',' ' ')
    NODES="${PARAM_ARRAY[4]}"
    MAX_DISTS="${PARAM_ARRAY[5]}"
    TEST_TYPE="${PARAM_ARRAY[6]}"
    SEED="${PARAM_ARRAY[7]}"

    FILENAME="output.txt"
    TMPFILE=$(mktemp)
        
    echo "Running $TEST_TYPE..."

    # Run the Python script with the specified parameters and append the output to TMPFILE
    { time $PY_ALIAS $SCRIPT \
        --nodes=$NODES \
        --max_dists=$MAX_DISTS \
        --optimizer=$TEST_TYPE \
        --filename=$FILENAME \
        --t_coh $T_COH \
        --p_gen $P_GEN \
        --p_swap=$P_SWAP \
        --w0 $W0 \
        --seed=$SEED \
    ; } 2>&1 | tee -a "$TMPFILE"

    # Extract the time taken and append it to the output file
    echo "Time taken:" >> "$FILENAME"
    tail -n 3 "$TMPFILE" >> "$FILENAME"
    rm "$TMPFILE"

    # Create a folder for the results if it doesn't exist
    RESULT_DIR="$GENERAL_RESULT_DIR/results_${TEST_TYPE}_tcoh${T_COH}_pgen${P_GEN}_pswap${P_SWAP}_w0${W0}_nodes${NODES}_maxdists${MAX_DISTS}_$SEED"
    mkdir -p "$RESULT_DIR"

    # Move the output file and the plots to the results folder
    mv "$FILENAME" "$RESULT_DIR/"
    if ls *${TEST_TYPE}.pdf 1> /dev/null 2>&1; then
        mv *${TEST_TYPE}.pdf "$RESULT_DIR/"
    else
        echo "No plots yielded for optimizer $TEST_TYPE"
    fi
done

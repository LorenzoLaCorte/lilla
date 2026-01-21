
#!/usr/bin/env bash
# gp_bell.sh
#
# Batch runner for Bell-diagonal (BellState) simulations using GP/BF methods.
#
# Usage:
#   chmod +x ./src/scripts/gp_bell.sh
#   ./src/scripts/gp_bell.sh
#
# Notes:
# - The sets below use the format:
#     depolarizing_rate   dephasing_rate   p_gen   p_swap   [l0,l1,l2,l3]   nodes   beta(max_dists)   optimizer(gp|bf)   seed(-1 for None)
# - If seed == -1, we omit --seed so your Python uses its default (None).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Point this to the Python file containing the updated __main__ shown in your message.
# If you prefer modules, replace with: PY_ENTRY=( "$PYTHON_BIN" -m src.optimization.<module_name> )
PY_SCRIPT="${PY_SCRIPT:-$ROOT_DIR/src/optimization/gp_asymmetric_protocols.py}"

OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/bell_gp}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Global defaults (override by exporting env vars)
GP_SHOTS="${GP_SHOTS:-100}"
GP_INITIAL_POINTS="${GP_INITIAL_POINTS:-10}"
CDF_THRESHOLD="${CDF_THRESHOLD:-0.99}"

# --------------------------------------------------------------------------------------
# Homogeneous Bell-diagonal regimes (examples you provided)
HOMOGENEOUS_SETS=(
  # ------------------------------------------------------------------ distribution at a given distance (Sec.V.A.)
  #                                                                    bruteforce simulations (Table 2)
  "0.00000027         0.00000027      0.000000096 0.85    [0.52, 0.12, 0.12, 0.12]        2      2          bf        -1"     # A, Beta=2
  "0.00000054         0.00000054      0.000015    0.85    [0.925, 0.025, 0.025, 0.025]    3      2          bf        -1"     # B, Beta=2
  "0.00000108         0.00000108      0.00092     0.85    [0.964, 0.009, 0.009, 0.009]    5      1          bf        -1"     # C, Beta=1
  #                                                                    bayesian opt. simulations (Table 2)
  "0.00000007         0.00000007      0.00092     0.85    [0.964, 0.009, 0.009, 0.009]    5      1          gp        42"     # C, Beta=1
  "0.00000007         0.00000007      0.00092     0.85    [0.964, 0.009, 0.009, 0.009]    5      2          gp        42"     # C, Beta=2
  "0.00000003         0.00000003      0.0026      0.85    [0.964, 0.009, 0.009, 0.009]    11     2          gp        42"     # D, Beta=2

  # ------------------------------------------------------------------ impact of distillation (Sec.V.B.)
  "0.00000007         0.00000007      0.00092     0.85    [0.94, 0.02, 0.02, 0.02]        5      0          bf        -1"     # C, Beta=0
  "0.00000007         0.00000007      0.00092     0.85    [0.955, 0.015, 0.015, 0.015]    5      0          bf        -1"     # C, Beta=0
  "0.00000007         0.00000007      0.00092     0.85    [0.97, 0.01, 0.01, 0.01 ]       5      0          bf        -1"     # C, Beta=0
  "0.00000007         0.00000007      0.00092     0.85    [0.985, 0.005, 0.005, 0.005 ]   5      0          bf        -1"     # C, Beta=0
  "0.00000007         0.00000007      0.00092     0.85    [0.94, 0.02, 0.02, 0.02 ]       5      3          gp        42"     # C, Beta=3
  "0.00000007         0.00000007      0.00092     0.85    [0.955, 0.015, 0.015, 0.015 ]   5      3          gp        42"     # C, Beta=3
  "0.00000007         0.00000007      0.00092     0.85    [0.97, 0.01, 0.01, 0.01 ]       5      3          gp        42"     # C, Beta=3
  "0.00000007         0.00000007      0.00092     0.85    [0.985, 0.005, 0.005, 0.005 ]   5      3          gp        42"     # C, Beta=3
)

# --------------------------------------------------------------------------------------

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: PY_SCRIPT not found at: $PY_SCRIPT"
  echo "Set PY_SCRIPT to your actual driver path, e.g.:"
  echo "  export PY_SCRIPT=\$ROOT_DIR/src/optimization/<your_file>.py"
  exit 1
fi

parse_set() {
  # Prints: depol deph pgen pswap l0 l1 l2 l3 nodes beta optimizer seed
  "$PYTHON_BIN" - "$1" <<'PY'
import ast, re, sys

s = sys.argv[1].strip()
pat = r'^\s*([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s+(\[.*\])\s+(\d+)\s+(\d+)\s+([A-Za-z]+)\s+(-?\d+)\s*$'
m = re.match(pat, s)
if not m:
    raise SystemExit(f"Cannot parse set line:\n{s}\nExpected: depol deph p_gen p_swap [l0,l1,l2,l3] nodes beta optimizer seed")

depol = float(m.group(1))
deph  = float(m.group(2))
pgen  = float(m.group(3))
pswap = float(m.group(4))
lambdas = ast.literal_eval(m.group(5))
if not (isinstance(lambdas, (list, tuple)) and len(lambdas) == 4):
    raise SystemExit(f"--lambdas must have 4 entries, got: {lambdas}")

nodes = int(m.group(6))
beta  = int(m.group(7))
opt   = m.group(8).lower()
seed  = int(m.group(9))

print(depol, deph, pgen, pswap, lambdas[0], lambdas[1], lambdas[2], lambdas[3], nodes, beta, opt, seed)
PY
}

sanitize() {
  # make a filename-safe token
  echo "$1" | tr ' /[](),:' '_' | tr -s '_' | sed 's/^_//;s/_$//'
}

run_one() {
  local set_line="$1"

  read -r depol deph pgen pswap l0 l1 l2 l3 nodes beta opt seed < <(parse_set "$set_line")

  local lam_tag
  lam_tag="$(sanitize "${l0}-${l1}-${l2}-${l3}")"

  local base="bell_N${nodes}_beta${beta}_pgen${pgen}_pswap${pswap}_depol${depol}_deph${deph}_lam${lam_tag}_${opt}"
  if [[ "$seed" -ge 0 ]]; then
    base="${base}_seed${seed}"
  fi
  base="$(sanitize "$base")"

  local out_json="$OUT_DIR/${base}.json"
  local log_file="$LOG_DIR/${base}.log"

  cmd=(
    "$PYTHON_BIN" "$PY_SCRIPT"
    --state bell
    --nodes "$nodes"
    --max_dists "$beta"
    --optimizer "$opt"
    --cdf_threshold "$CDF_THRESHOLD"
    --p_swap "$pswap"
    --p_gen "$pgen"
    --lambdas "$l0" "$l1" "$l2" "$l3"
    --depolarizing_rate "$depol"
    --dephasing_rate "$deph"
    --filename "$out_json"
  )

  if [[ "$opt" == "gp" ]]; then
    cmd+=( --gp_shots "$GP_SHOTS" --gp_initial_points "$GP_INITIAL_POINTS" )
  fi

  if [[ "$seed" -ge 0 ]]; then
    cmd+=( --seed "$seed" )
  fi

  echo "================================================================================"
  echo "SET: $set_line"
  echo "OUT: $out_json"
  echo "LOG: $log_file"
  echo "CMD: ${cmd[*]}"
  echo "================================================================================"

  "${cmd[@]}" 2>&1 | tee "$log_file"
}

for set_line in "${HOMOGENEOUS_SETS[@]}"; do
  run_one "$set_line"
done

# ------------------------------------------------------------------------------------------------------------------------------------
# Heterogeneous Bell-diagonal Chains:
# - p_gen: list of link-specific generation probabilities (length = nodes-1)
# - p_swap: scalar swap probability
# - lambdas: list of link-specific Bell-diagonal weights, one 4-tuple per link
#           i.e. for S=nodes-1 links we provide S items like: [l0,l1,l2,l3]
# - depolarizing_rate: scalar or list (depending on your simulator interpretation)
# - dephasing_rate: scalar or list (optional)
# - nodes: number of nodes in the chain
# - max_dists: maximum distillations applied to any link at any level
# ------------------------------------------------------------------------------------------------------------------------------------
#   (p_gen,                   p_swap,  lambdas_per_link,                                      depol,      deph,     nodes, max_dists, test_type, seed)
# ------------------------------------------------------------------------------------------------------------------------------------
HETEROGENEOUS_BELL_SETS=(
  # 3 links (nodes=4): provide 3 lambda-4tuples
  "[0.0025,0.0025,0.0025]     0.85     [[0.964,0.009,0.009,0.009],[0.964,0.009,0.009,0.009],[0.964,0.009,0.009,0.009]]   0.00000027 0.00000027 4 2 bf -1"
  "[0.0025,0.0025,0.00025]    0.85     [[0.964,0.009,0.009,0.009],[0.964,0.009,0.009,0.009],[0.94,0.02,0.02,0.02]]       0.00000054 0.00000054 4 2 bf -1"
  "[0.0025,0.0025,0.0025]     0.85     [[0.97,0.01,0.01,0.01],[0.964,0.009,0.009,0.009],[0.94,0.02,0.02,0.02]]           0.00000108 0.00000108 4 2 bf -1"
)

for PARAMETERS in "${HETEROGENEOUS_BELL_SETS[@]}"; do
  # Split by whitespace (safe because lambdas block contains no spaces)
  IFS=' ' read -r -a PARAM_ARRAY <<< "$PARAMETERS"

  P_GEN_RAW="${PARAM_ARRAY[0]}"
  P_SWAP="${PARAM_ARRAY[1]}"
  LAMBDAS_RAW="${PARAM_ARRAY[2]}"
  DEPOL="${PARAM_ARRAY[3]}"
  DEPH="${PARAM_ARRAY[4]}"
  NODES="${PARAM_ARRAY[5]}"
  MAX_DISTS="${PARAM_ARRAY[6]}"
  TEST_TYPE="${PARAM_ARRAY[7]}"
  SEED="${PARAM_ARRAY[8]}"

  # Clean p_gen list -> "0.1 0.2 0.3"
  P_GEN=$(echo "$P_GEN_RAW" | sed 's/\[//g' | sed 's/\]//g' | tr ',' ' ')

  # Convert lambdas JSON-ish into repeated CLI args:
  #   [[a,b,c,d],[e,f,g,h],...] -> "--lambdas a b c d --lambdas e f g h ..."
  # We use python for robust parsing (no jq dependency).
  LAMBDAS_ARGS="$($PY_ALIAS - <<PY
import ast
s = """$LAMBDAS_RAW"""
L = ast.literal_eval(s)
if not isinstance(L, (list, tuple)) or len(L) == 0:
    raise SystemExit("lambdas_per_link must be a non-empty list")
out = []
for item in L:
    if not isinstance(item, (list, tuple)) or len(item) != 4:
        raise SystemExit(f"Each lambdas entry must be length-4, got: {item}")
    out += ["--lambdas"] + [str(x) for x in item]
print(" ".join(out))
PY
)"

  FILENAME="output.txt"
  TMPFILE=$(mktemp)

  echo "Running Bell-heterogeneous $TEST_TYPE..."

  # Build command
  CMD=(
    $PY_ALIAS $SCRIPT
    --state=bell
    --nodes=$NODES
    --max_dists=$MAX_DISTS
    --optimizer=$TEST_TYPE
    --filename=$FILENAME
    --cdf_threshold=$CDF_THRESHOLD
    --p_gen $P_GEN
    --p_swap=$P_SWAP
    --depolarizing_rate=$DEPOL
    --dephasing_rate=$DEPH
  )

  # GP-specific knobs if you want them consistent with the homogeneous GP runs
  if [[ "$TEST_TYPE" == "gp" ]]; then
    CMD+=( --gp_shots=$GP_SHOTS --gp_initial_points=$GP_INITIAL_POINTS )
  fi

  # Seed: omit when -1
  if [[ "$SEED" -ge 0 ]]; then
    CMD+=( --seed=$SEED )
  fi

  # Append the per-link lambdas args (word-split intentionally)
  # shellcheck disable=SC2206
  CMD+=( $LAMBDAS_ARGS )

  { time "${CMD[@]}" ; } 2>&1 | tee -a "$TMPFILE"

  echo "Time taken:" >> "$FILENAME"
  tail -n 3 "$TMPFILE" >> "$FILENAME"
  rm "$TMPFILE"

  # Create a folder for the results
  # Make folder names safe-ish (strip spaces and brackets)
  SAFE_PGEN=$(echo "$P_GEN_RAW" | tr -d '[],' | tr ' ' '_')
  SAFE_LAM=$(echo "$LAMBDAS_RAW" | tr -d '[],' | tr ' ' '_' | tr -s '_')
  RESULT_DIR="$GENERAL_RESULT_DIR/results_${TEST_TYPE}_bell_pgen${SAFE_PGEN}_pswap${P_SWAP}_depol${DEPOL}_deph${DEPH}_lam${SAFE_LAM}_nodes${NODES}_maxdists${MAX_DISTS}_seed${SEED}"
  mkdir -p "$RESULT_DIR"

  mv "$FILENAME" "$RESULT_DIR/"

  if ls *${TEST_TYPE}.pdf 1> /dev/null 2>&1; then
    mv *${TEST_TYPE}.pdf "$RESULT_DIR/"
  else
    echo "No plots yielded for optimizer $TEST_TYPE"
  fi
done

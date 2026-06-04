#!/bin/bash
# Batch all training grid experiments (68 total) — local venv / tmux overnight runs.
#
# Suite 1 (8):  4 training types × {quasi_static, dynamic}
# Suite 2 (60): 4 training types × 5 (agents, obstacles) configs (sum=5) × 3 noise levels, always dynamic
#
# Local overnight (default — no flags needed):
#   tmux new -s grid
#   cd /local1/hishamb/cbf-rl-environment
#   CUDA_VISIBLE_DEVICES=3 ./submit_training_grid_venv.sh
#
# Preview the grid:
#   ./submit_training_grid_venv.sh --dry-run
#
# Venv already active (skip activate):
#   SKIP_ENV_SETUP=1 CUDA_VISIBLE_DEVICES=3 ./submit_training_grid_venv.sh
#
# Run one suite only:
#   ./submit_training_grid_venv.sh --suite 1
#
# Optional Hyak SLURM (not the default):
#   export HYAK_REPO_ROOT=/gscratch/scrubbed/jang1601/cbf-rl-environment
#   export VENV_PATH=/gscratch/scrubbed/jang1601/cbf-rl-venv
#   export SLURM_ACCOUNT=stf
#   ./submit_training_grid_venv.sh --submit
#
# obs_layout is NOT passed — train.py defaults to legacy and writes it to env_meta.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/cbf_env}"
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-0}"
GRID_LOG_ROOT="${GRID_LOG_ROOT:-${REPO_ROOT}/logs/grid_runs}"

# Hyak-only overrides (ignored for normal local runs).
HYAK_REPO_ROOT="${HYAK_REPO_ROOT:-/gscratch/scrubbed/jang1601/cbf-rl-environment}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu-2080ti}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_CPUS="${SLURM_CPUS:-4}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_TIME="${SLURM_TIME:-48:00:00}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || [[ "${USE_HYAK_PATHS:-0}" == "1" ]]; then
    REPO_ROOT="$HYAK_REPO_ROOT"
    GRID_LOG_ROOT="${HYAK_REPO_ROOT}/logs/grid_runs"
fi

TRAIN_TYPES=(naive cbf filter_only reward_only)

declare -a EXPERIMENT_NAMES=()
declare -a EXPERIMENT_CMDS=()

train_script() {
    case "$1" in
        naive)        echo "./train_naive.sh" ;;
        cbf)          echo "./train_cbf.sh" ;;
        filter_only)  echo "./train_filter_only.sh" ;;
        reward_only)  echo "./train_reward_only.sh" ;;
        *)
            echo "Unknown training type: $1" >&2
            return 1
            ;;
    esac
}

add_experiment() {
    local name="$1"
    shift
    EXPERIMENT_NAMES+=("$name")
    EXPERIMENT_CMDS+=("$*")
}

# --- Suite 1: dynamics × training type (8) ---
for dynamics in quasi_static dynamic; do
    for train_type in "${TRAIN_TYPES[@]}"; do
        script="$(train_script "$train_type")"
        add_experiment \
            "s1_${train_type}_${dynamics}" \
            "$script" --headless --dynamics_model "$dynamics"
    done
done

# --- Suite 2: dynamic × training type × agents/obstacles grid × noise (60) ---
AGENT_OBSTACLE_CONFIGS=(
    "1 4"
    "2 3"
    "3 2"
    "4 1"
    "5 0"
)

NOISE_LEVELS=(low medium high)

for train_type in "${TRAIN_TYPES[@]}"; do
    script="$(train_script "$train_type")"
    for config in "${AGENT_OBSTACLE_CONFIGS[@]}"; do
        read -r num_agents num_obstacles <<< "$config"
        for noise in "${NOISE_LEVELS[@]}"; do
            add_experiment \
                "s2_${train_type}_dynamic_a${num_agents}_o${num_obstacles}_n${noise}" \
                "$script" --headless --dynamics_model dynamic \
                --num_agents "$num_agents" \
                --num_obstacles "$num_obstacles" \
                --control_noise "$noise"
        done
    done
done

NUM_EXPERIMENTS="${#EXPERIMENT_CMDS[@]}"
if [[ "$NUM_EXPERIMENTS" -ne 68 ]]; then
    echo "Internal error: expected 68 experiments, got $NUM_EXPERIMENTS" >&2
    exit 1
fi

suite_range() {
    case "${1:-all}" in
        1) echo "0-7" ;;
        2) echo "8-67" ;;
        all) echo "0-67" ;;
        *)
            echo "Unknown suite: $1 (use 1, 2, or all)" >&2
            exit 1
            ;;
    esac
}

print_experiments() {
    local start="${1:-0}"
    local end="${2:-$((NUM_EXPERIMENTS - 1))}"
    for ((i = start; i <= end; i++)); do
        printf '[%02d] %s\n  %s\n\n' "$i" "${EXPERIMENT_NAMES[$i]}" "${EXPERIMENT_CMDS[$i]}"
    done
}

setup_env() {
    cd "$REPO_ROOT"
    if [[ "$SKIP_ENV_SETUP" == "1" ]]; then
        return 0
    fi
    if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
        echo "Venv not found at: ${VENV_PATH}/bin/activate" >&2
        echo "Set VENV_PATH to your venv directory, e.g.:" >&2
        echo "  export VENV_PATH=${SCRIPT_DIR}/cbf_env" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
}

run_experiment() {
    local idx="$1"
    local total="$2"
    local log_dir="$3"
    local exp_log="${log_dir}/$(printf '%02d' "$idx")_${EXPERIMENT_NAMES[$idx]}.log"

    echo ""
    echo "=== Experiment [$idx/$((total - 1))] ${EXPERIMENT_NAMES[$idx]} ==="
    echo "Started:  $(date -Is)"
    echo "Command:  ${EXPERIMENT_CMDS[$idx]}"
    echo "Repo:     $REPO_ROOT"
    echo "Python:   $(command -v python)"
    echo "GPU:      ${CUDA_VISIBLE_DEVICES:-all visible}"
    echo "Log file: $exp_log"
    echo ""

    {
        echo "=== Experiment [$idx] ${EXPERIMENT_NAMES[$idx]} ==="
        echo "Started: $(date -Is)"
        echo "Command: ${EXPERIMENT_CMDS[$idx]}"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
        echo ""
        # shellcheck disable=SC2086
        eval "${EXPERIMENT_CMDS[$idx]}"
        echo ""
        echo "Finished: $(date -Is)"
    } 2>&1 | tee "$exp_log"
}

MODE="local"
SUITE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  MODE="dry-run"; shift ;;
        --local)    MODE="local"; shift ;;
        --submit)   MODE="submit"; shift ;;
        --run-job)  MODE="run-job"; shift ;;
        --suite)    SUITE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,28p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ARRAY_RANGE="$(suite_range "$SUITE")"
IFS='-' read -r RANGE_START RANGE_END <<< "$ARRAY_RANGE"

case "$MODE" in
    dry-run)
        echo "Training grid ($((RANGE_END - RANGE_START + 1)) experiments, suite=$SUITE):"
        echo "Repo:  $REPO_ROOT"
        echo "Venv:  $VENV_PATH"
        echo "GPU:   ${CUDA_VISIBLE_DEVICES:-all visible}"
        print_experiments "$RANGE_START" "$RANGE_END"
        ;;
    local)
        setup_env
        GRID_SESSION_DIR="${GRID_LOG_ROOT}/grid_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$GRID_SESSION_DIR"
        MASTER_LOG="${GRID_SESSION_DIR}/master.log"

        {
            echo "Grid run started: $(date -Is)"
            echo "Repo: $REPO_ROOT"
            echo "Venv: $VENV_PATH"
            echo "Python: $(command -v python)"
            echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
            echo "Suite: $SUITE (experiments ${RANGE_START}-${RANGE_END})"
            echo "Per-experiment logs: $GRID_SESSION_DIR"
            echo ""
        } | tee "$MASTER_LOG"

        for ((i = RANGE_START; i <= RANGE_END; i++)); do
            run_experiment "$i" "$NUM_EXPERIMENTS" "$GRID_SESSION_DIR" | tee -a "$MASTER_LOG"
        done

        {
            echo ""
            echo "Grid run finished: $(date -Is)"
            echo "Logs: $GRID_SESSION_DIR"
        } | tee -a "$MASTER_LOG"
        ;;
    run-job)
        if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
            echo "SLURM_ARRAY_TASK_ID is not set" >&2
            exit 1
        fi
        REPO_ROOT="$HYAK_REPO_ROOT"
        setup_env
        GRID_SESSION_DIR="${GRID_LOG_ROOT}/slurm_${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID}"
        mkdir -p "$GRID_SESSION_DIR"
        run_experiment "$SLURM_ARRAY_TASK_ID" "$NUM_EXPERIMENTS" "$GRID_SESSION_DIR"
        ;;
    submit)
        if [[ -z "$SLURM_ACCOUNT" ]]; then
            echo "Set SLURM_ACCOUNT before submitting, e.g.:"
            echo "  export SLURM_ACCOUNT=your_account"
            exit 1
        fi
        JOB_SCRIPT="$SCRIPT_DIR/.submit_training_grid_venv_job.sh"
        cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=cbf-nav-grid-venv
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH --array=${ARRAY_RANGE}
#SBATCH --output=${HYAK_REPO_ROOT}/logs/slurm/cbf_nav_grid_venv_%A_%a.out
#SBATCH --error=${HYAK_REPO_ROOT}/logs/slurm/cbf_nav_grid_venv_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export USE_HYAK_PATHS=1
export VENV_PATH="${VENV_PATH}"
export HYAK_REPO_ROOT="${HYAK_REPO_ROOT}"
export SKIP_ENV_SETUP=0
bash "${HYAK_REPO_ROOT}/submit_training_grid_venv.sh" --run-job
EOF
        chmod +x "$JOB_SCRIPT"
        mkdir -p "${HYAK_REPO_ROOT}/logs/slurm" 2>/dev/null || true
        sbatch "$JOB_SCRIPT"
        echo "Submitted SLURM array jobs ${ARRAY_RANGE} (suite=$SUITE)."
        echo "Logs: ${HYAK_REPO_ROOT}/logs/slurm/cbf_nav_grid_venv_<jobid>_<task>.out"
        ;;
esac

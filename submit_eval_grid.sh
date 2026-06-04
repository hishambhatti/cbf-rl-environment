#!/bin/bash
# Evaluate all trained models discovered under logs/.
#
# Scans logs/ for every run directory containing env_meta.json + at least one
# model_*.pt checkpoint, reads the env type, and runs test.py --headless.
#
# Modes:
#   ./submit_eval_grid.sh --dry-run          # print what would run
#   SKIP_ENV_SETUP=1 ./submit_eval_grid.sh --local   # run sequentially (interactive GPU node)
#   ./submit_eval_grid.sh --submit           # SLURM array job
#
# On Hyak:
#   export SLURM_ACCOUNT=stf
#   ./submit_eval_grid.sh --submit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYAK_REPO_ROOT="${HYAK_REPO_ROOT:-/gscratch/scrubbed/jang1601/cbf-rl-environment}"
CONDA_ENV="${CONDA_ENV:-/gscratch/scrubbed/jang1601/cbf-rl}"
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-0}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-stf}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu-2080ti}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_CPUS="${SLURM_CPUS:-4}"
SLURM_MEM="${SLURM_MEM:-16G}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || [[ "${USE_HYAK_PATHS:-0}" == "1" ]]; then
    REPO_ROOT="$HYAK_REPO_ROOT"
else
    REPO_ROOT="$SCRIPT_DIR"
fi

LOGS_DIR="$REPO_ROOT/logs"
MANIFEST="$SCRIPT_DIR/.eval_manifest.tsv"

test_script() {
    case "$1" in
        naive)        echo "./test_naive.sh" ;;
        cbf)          echo "./test_cbf.sh" ;;
        filter_only)  echo "./test_filter_only.sh" ;;
        reward_only)  echo "./test_reward_only.sh" ;;
        *)            echo "./test_naive.sh" ;;
    esac
}

# ---------------------------------------------------------------------------
# Discover all run directories that have env_meta.json + a checkpoint
# ---------------------------------------------------------------------------
discover_runs() {
    : > "$MANIFEST"
    local count=0
    while IFS= read -r -d '' meta_file; do
        local run_dir
        run_dir="$(dirname "$meta_file")"
        # Must have at least one checkpoint
        if ! ls "$run_dir"/model_*.pt &>/dev/null; then
            continue
        fi
        local env_type
        env_type="$(python -c "
import json, sys
try:
    d = json.load(open('$meta_file'))
    print(d.get('env', 'naive'))
except Exception:
    print('naive')
")"
        printf '%s\t%s\n' "$env_type" "$run_dir" >> "$MANIFEST"
        count=$((count + 1))
    done < <(find "$LOGS_DIR" -maxdepth 4 -name "env_meta.json" -print0 2>/dev/null | sort -z)
    echo "$count"
}

setup_env() {
    if [[ "$SKIP_ENV_SETUP" == "1" ]]; then
        cd "$REPO_ROOT"
        return 0
    fi
    { conda deactivate 2>/dev/null || true; }
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL 2>/dev/null || true
    module load conda
    conda activate "$CONDA_ENV"
    cd "$REPO_ROOT"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODE="local"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  MODE="dry-run"; shift ;;
        --local)    MODE="local"; shift ;;
        --submit)   MODE="submit"; shift ;;
        --run-job)  MODE="run-job"; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
case "$MODE" in
    dry-run)
        setup_env
        echo "Discovering runs under $LOGS_DIR ..."
        NUM_RUNS="$(discover_runs)"
        echo "Found $NUM_RUNS trained runs:"
        idx=0
        while IFS=$'\t' read -r env_type run_dir; do
            printf '[%02d] env=%-12s  %s\n' "$idx" "$env_type" "$run_dir"
            idx=$((idx + 1))
        done < "$MANIFEST"
        ;;

    local)
        setup_env
        echo "Discovering runs under $LOGS_DIR ..."
        NUM_RUNS="$(discover_runs)"
        echo "Found $NUM_RUNS trained runs. Starting evaluation..."
        idx=0
        while IFS=$'\t' read -r env_type run_dir; do
            script="$(test_script "$env_type")"
            echo "=== Eval [$idx] env=$env_type ==="
            echo "Run dir: $run_dir"
            $script --load_run "$run_dir" --headless
            idx=$((idx + 1))
        done < "$MANIFEST"
        ;;

    submit)
        if [[ -z "$SLURM_ACCOUNT" ]]; then
            echo "Set SLURM_ACCOUNT before submitting." >&2
            exit 1
        fi
        setup_env
        echo "Discovering runs under $LOGS_DIR ..."
        NUM_RUNS="$(discover_runs)"
        if [[ "$NUM_RUNS" -eq 0 ]]; then
            echo "No trained runs found under $LOGS_DIR." >&2
            exit 1
        fi
        echo "Found $NUM_RUNS trained runs. Submitting SLURM array..."

        ARRAY_RANGE="0-$((NUM_RUNS - 1))"
        JOB_SCRIPT="$SCRIPT_DIR/.submit_eval_grid_job.sh"
        cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=cbf-eval-grid
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH --array=${ARRAY_RANGE}
#SBATCH --output=${HYAK_REPO_ROOT}/logs/slurm/cbf_eval_grid_%A_%a.out
#SBATCH --error=${HYAK_REPO_ROOT}/logs/slurm/cbf_eval_grid_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
{ conda deactivate 2>/dev/null || true; }
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL 2>/dev/null || true
module load conda
conda activate ${CONDA_ENV}
cd ${HYAK_REPO_ROOT}
export SKIP_ENV_SETUP=1
bash "${HYAK_REPO_ROOT}/submit_eval_grid.sh" --run-job
EOF
        chmod +x "$JOB_SCRIPT"
        mkdir -p "${HYAK_REPO_ROOT}/logs/slurm" 2>/dev/null || true
        sbatch "$JOB_SCRIPT"
        echo "Submitted SLURM eval array jobs ${ARRAY_RANGE}."
        echo "Logs: ${HYAK_REPO_ROOT}/logs/slurm/cbf_eval_grid_<jobid>_<task>.out"
        ;;

    run-job)
        if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
            echo "SLURM_ARRAY_TASK_ID is not set" >&2
            exit 1
        fi
        REPO_ROOT="$HYAK_REPO_ROOT"
        setup_env

        if [[ ! -f "$MANIFEST" ]]; then
            echo "Manifest not found: $MANIFEST" >&2
            exit 1
        fi

        IDX="$SLURM_ARRAY_TASK_ID"
        line="$(sed -n "$((IDX + 1))p" "$MANIFEST")"
        env_type="$(echo "$line" | cut -f1)"
        run_dir="$(echo "$line" | cut -f2)"

        script="$(test_script "$env_type")"
        echo "=== Eval [$IDX] env=$env_type ==="
        echo "Run dir: $run_dir"
        $script --load_run "$run_dir" --headless
        ;;
esac

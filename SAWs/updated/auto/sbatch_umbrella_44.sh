#!/bin/bash
#SBATCH --job-name=Umbrella44
#SBATCH --output=Umbrella44_%j.out
#SBATCH --error=err_%j.err
#SBATCH --partition=tokmakoff
#SBATCH --account=pi-tokmakoff
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25       # = the window count printed by --show_window_design
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=36:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load python/anaconda-2021.05

# One BLAS thread per process: this job forks one walker per window and they
# would otherwise oversubscribe the node many times over.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

which python
python --version
python -c "import numpy; print('numpy', numpy.__version__)"

if [[ ! -f single_chain_umbrella_sampling.py ]]; then
    echo "Submit this job from SAWs/updated/auto." >&2
    exit 1
fi

# ---------------------------------------------------------------- settings --
MODE="production"                 # "pilot" or "production"
N=44
TAIL_SLOPE=6.0                    # |d log P0/dm| at m_max; see README step 2
UMBRELLA_K=0.30
WINDOW_SPACING=3

RUN_DIR="umbrella_runs/N${N}"
CKPT="$RUN_DIR/umbrella_checkpoint.npz"
OUT="$RUN_DIR/RnBaseline${N}_umbrella.npz"
mkdir -p "$RUN_DIR"

# --steps_per_window MUST be the final target on the very first submission.
# burn_steps = round(burnin * steps_per_window) is computed once and frozen in
# the checkpoint; raising the target later keeps the old burn boundary and only
# prints a note. Wall-time checkpointing is what splits the work across jobs,
# not a smaller target.
STEPS_PER_WINDOW=60000000

# Leave the sampler 15 min inside the SLURM limit to write its final checkpoint.
WALL_SECONDS=$((36 * 3600))
MARGIN_SECONDS=900

if [[ "$MODE" == "pilot" ]]; then
    # Gates off so the run always completes and writes its diagnostics, which
    # is the whole point of the pilot: it measures the real tail slope.
    STEPS_PER_WINDOW=4000000
    OUT="$RUN_DIR/pilot_N${N}.npz"
    CKPT="$RUN_DIR/pilot_checkpoint.npz"
    GATES=(--min_samples_per_level 0 --min_adjacent_overlap 0.0
           --min_swap_acceptance 0.0 --min_round_trips 0
           --max_level_rel_stderr 0.0)
else
    GATES=(--min_samples_per_level 1000 --min_adjacent_overlap 0.10
           --min_swap_acceptance 0.10 --min_round_trips 1
           --max_level_rel_stderr 0.25)
fi

# ------------------------------------------------------------------ resume --
RESUME_ARGS=()
if [[ -f "$CKPT" ]]; then
    echo "Found checkpoint $CKPT - resuming."
    RESUME_ARGS+=(--resume_checkpoint "$CKPT")
else
    echo "No checkpoint found - starting fresh."
fi

# Print the ladder before committing a node to it.
python -u single_chain_umbrella_sampling.py \
    --N "$N" --tail_slope "$TAIL_SLOPE" \
    --umbrella_k "$UMBRELLA_K" --window_spacing "$WINDOW_SPACING" \
    --show_window_design

# --n_processes is omitted on purpose: it defaults to SLURM_CPUS_PER_TASK,
# capped at the window count.
python -u single_chain_umbrella_sampling.py \
    --N "$N" \
    --tail_slope "$TAIL_SLOPE" \
    --umbrella_k "$UMBRELLA_K" \
    --window_spacing "$WINDOW_SPACING" \
    --steps_per_window "$STEPS_PER_WINDOW" \
    --burnin 0.30 \
    --sample_every 500 \
    --exchange_every 1000 \
    --pull_move_weight 0.25 \
    --init auto \
    --base_seed 53 \
    --rg_bins 60 \
    --n_blocks 20 \
    --checkpoint "$CKPT" \
    --checkpoint_every_seconds 1800 \
    --max_wall_seconds "$WALL_SECONDS" \
    --shutdown_margin_seconds "$MARGIN_SECONDS" \
    --output "$OUT" \
    "${GATES[@]}" \
    "${RESUME_ARGS[@]}"

# If the job stopped on wall time it printed "Run segment checkpointed
# successfully" and wrote no baseline. Just resubmit this same script; it
# resumes from the checkpoint. Repeat until "DIST_FILE = ..." appears.

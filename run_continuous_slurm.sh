#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=124GB
#SBATCH --output=results/%j/slurm-output.out

module load mamba
source activate env-bridge

mkdir -p results
JOB_DIR="results/${SLURM_JOB_ID}"
mkdir -p $JOB_DIR

monitor_processes() {
    while true; do
        echo "=== $(date) ===" >> $JOB_DIR/process_monitor.log
        echo "Process count: $(pgrep -u $USER python | wc -l)" >> $JOB_DIR/process_monitor.log
        echo "Thread count: $(ps -eLf | grep python | grep -v grep | wc -l)" >> $JOB_DIR/process_monitor.log
        echo "CPU frequencies (MHz):" >> $JOB_DIR/process_monitor.log
        cat /proc/cpuinfo | grep "cpu MHz" | head -10 >> $JOB_DIR/process_monitor.log
        echo "Threading settings: OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS" >> $JOB_DIR/process_monitor.log
        echo "Load average: $(cat /proc/loadavg)" >> $JOB_DIR/process_monitor.log
        echo "Memory usage by process:" >> $JOB_DIR/process_monitor.log
        ps -u $USER -o pid,ppid,pcpu,pmem,rss,vsz,comm --sort=-rss | head -25 >> $JOB_DIR/process_monitor.log
        echo "Memory info:" >> $JOB_DIR/process_monitor.log
        free -h >> $JOB_DIR/process_monitor.log
        echo "" >> $JOB_DIR/process_monitor.log
        sleep 30
    done
}
monitor_processes &
MONITOR_PID=$!

# WITHOUT THIS, MUCH SLOWER (disable threading)
export OMP_NUM_THREADS=1           # OpenMP
export MKL_NUM_THREADS=1           # Intel MKL 
export OPENBLAS_NUM_THREADS=1      # OpenBLAS
export NUMEXPR_NUM_THREADS=1       # NumExpr
export VECLIB_MAXIMUM_THREADS=1    # Apple/ARM systems
export BLIS_NUM_THREADS=1          # AMD BLIS
export GOTO_NUM_THREADS=1          # GotoBLAS

echo "Node: $(hostname)" >> $JOB_DIR/slurm-output.out
echo "CPUs available: $(nproc)" >> $JOB_DIR/slurm-output.out
echo "Memory available: $(free -h)" >> $JOB_DIR/slurm-output.out
echo "Running command: python -m main_mujoco $@" >> $JOB_DIR/slurm-output.out
echo "Job ID: $SLURM_JOB_ID" >> $JOB_DIR/slurm-output.out
echo "Start time: $(date)" >> $JOB_DIR/slurm-output.out
echo "Runtime limit: $(scontrol show job $SLURM_JOB_ID | grep TimeLimit | awk '{print $2}')" >> $JOB_DIR/slurm-output.out

python -m main_continuous "$@"  # passing all cmds to python module

# cleanup
kill $MONITOR_PID 2>/dev/null
echo "End time: $(date)" >> $JOB_DIR/slurm-output.out
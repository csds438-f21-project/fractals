#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p classc
#SBATCH -A sxg125_csds438
#SBATCH --mem=8gb
#SBATCH --time=12:00:00
#SBATCH -J lewicki 

# Copy required files to scratch dir
cp generate_fractal.py $PFSDIR
cp -r lewicki/ $PFSDIR
cd $PFSDIR

# Run
echo "Running lewicki"
python3 generate_fractal.py lewicki 1 2 4 8 16 32

# Copy data back to submit dir
mkdir -p $SLURM_SUBMIT_DIR/lewicki_frames/
cp -r frames/ $SLURM_SUBMIT_DIR/lewicki_frames/

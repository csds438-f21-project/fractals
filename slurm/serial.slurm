#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p classc
#SBATCH -A sxg125_csds438
#SBATCH --mem=8gb
#SBATCH --time=12:00:00
#SBATCH -J serial

# Copy required files to scratch dir
cp generate_fractal.py $PFSDIR
cd $PFSDIR

# Run
echo "Running serial baseline"
python3 generate_fractal.py serial

# Copy data back to submit dir
mkdir -p $SLURM_SUBMIT_DIR/serial_frames/
cp -r frames/ $SLURM_SUBMIT_DIR/serial_frames/

#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J testjob
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 3:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s233668@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# Debug: Check the values of UNO and DOS
#echo "Before exporting: UNO=${UNO}, DOS=${DOS}" >> debug.log

# Set variables (ensure they are set dynamically when submitting the job)
LR=${LR:-0.0001}  # Default to 0.0001 if learning rate is not set
BS=${BS:-8}  # Default to 8 if batch size is not set
EP=${EP:-30}  # Default to 30 if epochs is not set
WD=${WD:-0}  # Default to 0 if weight decay is not set
HEAD=${HEAD:-6}  # Default to 6 if head dim is not set
export LR
export BS
export EP
export WD
export HEAD

# Debug: Confirm export worked
#echo "After exporting: UNO=${UNO}, DOS=${DOS}" >> debug.log

nvidia-smi
# Load the cuda module
#module load cuda/11.6

#/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery
python3 hpctest.py "$LR" "$BS" "$EP" "$WD" "$HEAD" > outputtest.txt

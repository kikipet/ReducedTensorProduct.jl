#!/bin/bash
#SBATCH -n 6
#SBATCH -N 6

source /etc/profile
cd /home/gridsan/sekim

module load julia/1.6.1
cd ReducedTensorProduct.jl
julia -p 6 benchmark.jl

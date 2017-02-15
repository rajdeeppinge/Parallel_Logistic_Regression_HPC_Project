#!/bin/bash

gcc -Wall -fopenmp -o mlp_hpc_multiclass_swap mlp_hpc_multiclass_swap.c -lgsl -lgslcblas -lm
gcc -Wall -fopenmp -o mlp_hpc_multiclass_parallel_swap mlp_hpc_multiclass_parallel_swap.c -lgsl -lgslcblas -lm

echo changing input size
echo

export OMP_NUM_THREADS=12
input=10
#size="$(echo "l($input)/l(10)" | bc -l)"
#echo number of points = 1e${size%%.*}
serial_line="$(./mlp_hpc_multiclass_swap mnist_large.txt mnist_y.txt 600 785 10 0.1 10)"
echo Serial: ${serial_line}
stime="$(echo "${serial_line}" | awk '{ print $2 }')"
parallel_line="$(./mlp_hpc_multiclass_parallel_swap mnist_large.txt mnist_y.txt 600 785 10 0.1 10)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${input}: ${speedup_multiclass_swap}
echo ${input} "$speedup_multiclass_swap" > iterationVsSpeedup.txt
echo 

sleep 2

for input in 100 1000
do

#input=60000
#size="$(echo "l($input)/l(10)" | bc -l)"
#echo number of points = 1e${size%%.*}
serial_line="$(./mlp_hpc_multiclass_swap mnist_large.txt mnist_y.txt 600 785 "${input}" 0.1 10)"
echo Serial: ${serial_line}
stime="$(echo "${serial_line}" | awk '{ print $2 }')"

sleep 2

parallel_line="$(./mlp_hpc_multiclass_parallel_swap mnist_large.txt mnist_y.txt 600 785 "${input}" 0.1 10)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${input}: ${speedup_multiclass_swap}
echo ${input} "$speedup_multiclass_swap" >> iterationVsSpeedup.txt
echo 

sleep 2

done

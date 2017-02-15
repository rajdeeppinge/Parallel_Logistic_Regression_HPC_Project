#!/bin/bash

gcc -Wall -fopenmp -o mlp_hpc_multiclass_swap mlp_hpc_multiclass_swap.c -lgsl -lgslcblas -lm
gcc -Wall -fopenmp -o mlp_hpc_multiclass_parallel_swap mlp_hpc_multiclass_parallel_swap.c -lgsl -lgslcblas -lm

echo changing number of threads, size = 7352 x 661
echo	   

thread=1
echo Threads_${thread}
export OMP_NUM_THREADS=${thread}
serial_line="$(./mlp_hpc_multiclass_swap X_train.dat y_train.txt 7352 661 100 0.1 6)"
echo Serial: ${serial_line}
stime="$(echo "${serial_line}" | awk '{ print $2 }')"
parallel_line="$(./mlp_hpc_multiclass_parallel_swap X_multi_test.dat y_multi_test.dat 100 401 1000 0.1 9)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${thread} thread: ${speedup_multiclass_swap}
echo "$thread $speedup_multiclass_swap" > multiclass_swap_coresVsSpeedup_size7352.txt
echo 

sleep 2

for i in 2 3 4
do

thread=$i
echo Threads_${thread}
export OMP_NUM_THREADS=${thread}
#serial_line="$(./mlp_hpc_multiclass_swap X_multi_input.dat y_multi_input.dat 4000 401 100 0.1)"
echo Serial: ${serial_line}
#stime="$(echo "${serial_line}" | awk '{ print $2 }')"
parallel_line="$(./mlp_hpc_multiclass_parallel_swap X_multi_test.dat y_multi_test.dat 100 401 1000 0.1 9)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${thread} thread: ${speedup_multiclass_swap}
echo "$thread $speedup_multiclass_swap" >> multiclass_swap_coresVsSpeedup_size100.txt
echo 

sleep 2

done
      
echo changing input size
echo

input=100
#size="$(echo "l($input)/l(10)" | bc -l)"
#echo number of points = 1e${size%%.*}
#serial_line="$(./mlp_hpc_multiclass_swap X_multi_input.dat y_multi_input.dat 4000 401 100 0.1)"
echo Serial: ${serial_line}
#stime="$(echo "${serial_line}" | awk '{ print $2 }')"
#parallel_line="$(./mlp_hpc_multiclass_parallel_swap X_multi_input.dat y_multi_input.dat 4000 401 100 0.1)"
echo Parallel: ${parallel_line}
#ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

#speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${input}: ${speedup_multiclass_swap}
echo ${input} "$stime $ptime" >> multiclass_swap_problemSizeVsTime.txt
echo ${input} "$speedup_multiclass_swap" >> multiclass_swap_problemSizeVsSpeedup.txt
echo 

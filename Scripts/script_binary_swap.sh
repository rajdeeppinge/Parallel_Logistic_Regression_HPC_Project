#!/bin/bash

gcc -Wall -fopenmp -o mlp_hpc_serial_swap mlp_hpc_serial_swap.c -lgsl -lgslcblas -lm
gcc -Wall -fopenmp -o mlp_hpc_parallel_swap_loops mlp_hpc_parallel_swap_loops.c -lgsl -lgslcblas -lm

echo changing number of threads, size = 1499 x 401
echo	   

sleep 2

thread=1
echo Threads_${thread}
export OMP_NUM_THREADS=${thread}
serial_line="$(./mlp_hpc_serial_swap X_bin_update.dat y_bin.dat 1499 401 1000 0.1)"
echo Serial: ${serial_line}
stime="$(echo "${serial_line}" | awk '{ print $2 }')"
parallel_line="$(./mlp_hpc_parallel_swap_loops X_bin_update.dat y_bin.dat 1499 401 1000 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${thread} thread: ${speedup_multiclass_swap}
echo "$thread $speedup_multiclass_swap" > binary_swap_coresVsSpeedup_size1499.txt
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
parallel_line="$(./mlp_hpc_parallel_swap_loops X_bin_update.dat y_bin.dat 1499 401 1000 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${thread} thread: ${speedup_multiclass_swap}
echo "$thread $speedup_multiclass_swap" >> binary_swap_coresVsSpeedup_size1499.txt
echo 

sleep 2

done
      
echo changing input size
echo

input=1499
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
echo ${input} "$stime $ptime" > binary_swap_problemSizeVsTime.txt
echo ${input} "$speedup_multiclass_swap" > binary_swap_problemSizeVsSpeedup.txt
echo 

sleep 2

for input in 10 100 1000
do

#input=60000
#size="$(echo "l($input)/l(10)" | bc -l)"
#echo number of points = 1e${size%%.*}
serial_line="$(./mlp_hpc_serial_swap X_bin_update.dat y_bin.dat "${input}" 401 1000 0.1)"
echo Serial: ${serial_line}
stime="$(echo "${serial_line}" | awk '{ print $2 }')"
parallel_line="$(./mlp_hpc_parallel_swap_loops X_bin_update.dat y_bin.dat "${input}" 401 1000 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${input}: ${speedup_multiclass_swap}
echo ${input} "$stime $ptime" >> binary_swap_problemSizeVsTime.txt
echo ${input} "$speedup_multiclass_swap" >> binary_swap_problemSizeVsSpeedup.txt
echo 

sleep 2

done

sleep 2

echo changing number of threads, size = 100 x 401
echo	   

sleep 2

thread=1
echo Threads_${thread}
export OMP_NUM_THREADS=${thread}
serial_line="$(./mlp_hpc_serial_swap X_bin_update.dat y_bin.dat 100 401 1000 0.1)"
echo Serial: ${serial_line}
stime="$(echo "${serial_line}" | awk '{ print $2 }')"
parallel_line="$(./mlp_hpc_parallel_swap_loops X_bin_update.dat y_bin.dat 100 401 1000 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${thread} thread: ${speedup_multiclass_swap}
echo "$thread $speedup_multiclass_swap" > binary_swap_coresVsSpeedup_size100.txt
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
parallel_line="$(./mlp_hpc_parallel_swap_loops X_bin_update.dat y_bin.dat 100 401 1000 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_multiclass_swap="$(echo "$stime/$ptime" | bc -l)"

echo speedup ${thread} thread: ${speedup_multiclass_swap}
echo "$thread $speedup_multiclass_swap" >> binary_swap_coresVsSpeedup_size100.txt
echo 

sleep 2

done

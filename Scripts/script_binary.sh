#!/bin/bash

gcc -Wall -fopenmp -o mlp_hpc mlp_hpc.c -lgsl -lgslcblas -lm
gcc -Wall -fopenmp -o mlp_hpc_inner_parallel_reduction mlp_hpc_inner_parallel_reduction.c -lgsl -lgslcblas -lm
gcc -Wall -fopenmp -o mlp_hpc_outer_parallel_collapse mlp_hpc_outer_parallel_collapse.c -lgsl -lgslcblas -lm

echo changing number of threads, size = 1499 x 401
echo	   

sleep 2

thread=1
echo Threads_${thread}
export OMP_NUM_THREADS=${thread}

serial_line="$(./mlp_hpc X_bin_update.dat y_bin.dat 1499 401 10 0.1)"
echo Serial: ${serial_line}
stime="$(echo "${serial_line}" | awk '{ print $2 }')"

parallel_line="$(./mlp_hpc_inner_parallel_reduction X_bin_update.dat y_bin.dat 1499 401 10 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_inner_parallel="$(echo "$stime/$ptime" | bc -l)"

parallel_outer="$(./mlp_hpc_outer_parallel_collapse X_bin_update.dat y_bin.dat 1499 401 10 0.1)"
echo Parallel Outer: ${parallel_outer}
poutertime="$(echo "${parallel_outer}" | awk '{ print $2 }')"

speedup_outer_parallel="$(echo "$stime/$poutertime" | bc -l)"

echo speedup ${thread} thread: ${speedup_inner_parallel} ${speedup_outer_parallel}
echo "$thread $speedup_inner_parallel $speedup_outer_parallel" > basic_coresVsSpeedup_size1499.txt
echo 

sleep 2

for i in 2 3 4
do

thread=$i
echo Threads_${thread}
export OMP_NUM_THREADS=${thread}

#serial_line="$(./mlp_hpc X_bin_update.dat y_bin.dat 1499 401 100 0.1)"
echo Serial: ${serial_line}
#stime="$(echo "${serial_line}" | awk '{ print $2 }')"

parallel_line="$(./mlp_hpc_inner_parallel_reduction X_bin_update.dat y_bin.dat 1499 401 10 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_inner_parallel="$(echo "$stime/$ptime" | bc -l)"

parallel_outer="$(./mlp_hpc_outer_parallel_collapse X_bin_update.dat y_bin.dat 1499 401 10 0.1)"
echo Parallel Outer: ${parallel_outer}
poutertime="$(echo "${parallel_outer}" | awk '{ print $2 }')"

speedup_outer_parallel="$(echo "$stime/$poutertime" | bc -l)"

echo speedup ${thread} thread: ${speedup_inner_parallel} ${speedup_outer_parallel}
echo "$thread $speedup_inner_parallel $speedup_outer_parallel" >> basic_coresVsSpeedup_size1499.txt
echo 

sleep 2

done
      
echo changing input size
echo

input=1499
#size="$(echo "l($input)/l(10)" | bc -l)"
#echo number of points = 1e${size%%.*}

#serial_line="$(./mlp_hpc X_bin_update.dat y_bin.dat 1499 401 100 0.1)"
echo Serial: ${serial_line}
#stime="$(echo "${serial_line}" | awk '{ print $2 }')"

#parallel_line="$(./mlp_hpc_inner_parallel_reduction X_bin_update.dat y_bin.dat 1499 401 100 0.1)"
echo Parallel: ${parallel_line}
#ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

#speedup_inner_parallel="$(echo "$stime/$ptime" | bc -l)"

#parallel_outer="$(./mlp_hpc_outer_parallel_collapse X_bin_update.dat y_bin.dat 1499 401 100 0.1)"
echo Parallel Outer: ${parallel_outer}
#poutertime="$(echo "${parallel_outer}" | awk '{ print $2 }')"

#speedup_outer_parallel="$(echo "$stime/$poutertime" | bc -l)"

echo speedup ${input}: ${speedup_inner_parallel} ${speedup_outer_parallel}
echo ${input} "$stime $ptime $poutertime" > basic_problemSizeVsTime.txt
echo ${input} "$speedup_inner_parallel $speedup_outer_parallel" > basic_problemSizeVsSpeedup.txt
echo 

sleep 2

for input in 10 100 1000
do

#size="$(echo "l($input)/l(10)" | bc -l)"
#echo number of points = 1e${size%%.*}

serial_line="$(./mlp_hpc X_bin_update.dat y_bin.dat "${input}" 401 10 0.1)"
echo Serial: ${serial_line}
stime="$(echo "${serial_line}" | awk '{ print $2 }')"

parallel_line="$(./mlp_hpc_inner_parallel_reduction X_bin_update.dat y_bin.dat "${input}" 401 10 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_inner_parallel="$(echo "$stime/$ptime" | bc -l)"

parallel_outer="$(./mlp_hpc_outer_parallel_collapse X_bin_update.dat y_bin.dat "${input}" 401 10 0.1)"
echo Parallel Outer: ${parallel_outer}
poutertime="$(echo "${parallel_outer}" | awk '{ print $2 }')"

speedup_outer_parallel="$(echo "$stime/$poutertime" | bc -l)"

echo speedup ${input}: ${speedup_inner_parallel} ${speedup_outer_parallel}
echo ${input} "$stime $ptime $poutertime" >> basic_problemSizeVsTime.txt
echo ${input} "$speedup_inner_parallel $speedup_outer_parallel" >> basic_problemSizeVsSpeedup.txt
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

serial_line="$(./mlp_hpc X_bin_update.dat y_bin.dat 100 401 10 0.1)"
echo Serial: ${serial_line}
stime="$(echo "${serial_line}" | awk '{ print $2 }')"

parallel_line="$(./mlp_hpc_inner_parallel_reduction X_bin_update.dat y_bin.dat 100 401 10 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_inner_parallel="$(echo "$stime/$ptime" | bc -l)"

parallel_outer="$(./mlp_hpc_outer_parallel_collapse X_bin_update.dat y_bin.dat 100 401 10 0.1)"
echo Parallel Outer: ${parallel_outer}
poutertime="$(echo "${parallel_outer}" | awk '{ print $2 }')"

speedup_outer_parallel="$(echo "$stime/$poutertime" | bc -l)"

echo speedup ${thread} thread: ${speedup_inner_parallel} ${speedup_outer_parallel}
echo "$thread $speedup_inner_parallel $speedup_outer_parallel" > basic_coresVsSpeedup_size100.txt
echo 

sleep 2

for i in 2 3 4
do

thread=$i
echo Threads_${thread}
export OMP_NUM_THREADS=${thread}

#serial_line="$(./mlp_hpc X_bin_update.dat y_bin.dat 1499 401 100 0.1)"
echo Serial: ${serial_line}
#stime="$(echo "${serial_line}" | awk '{ print $2 }')"

parallel_line="$(./mlp_hpc_inner_parallel_reduction X_bin_update.dat y_bin.dat 100 401 10 0.1)"
echo Parallel: ${parallel_line}
ptime="$(echo "${parallel_line}" | awk '{ print $2 }')"

speedup_inner_parallel="$(echo "$stime/$ptime" | bc -l)"

parallel_outer="$(./mlp_hpc_outer_parallel_collapse X_bin_update.dat y_bin.dat 100 401 10 0.1)"
echo Parallel Outer: ${parallel_outer}
poutertime="$(echo "${parallel_outer}" | awk '{ print $2 }')"

speedup_outer_parallel="$(echo "$stime/$poutertime" | bc -l)"

echo speedup ${thread} thread: ${speedup_inner_parallel} ${speedup_outer_parallel}
echo "$thread $speedup_inner_parallel $speedup_outer_parallel" >> basic_coresVsSpeedup_size100.txt
echo 

sleep 2

done

set terminal epslatex size 5,3 color colortext

set output 'basic_problemSizeVsTime.tex'
set title 'Basic Binary classification: Variation in time compared to the number of training samples iterations: 1000'
set xlabel 'Training Samples[log base 10]'
set ylabel 'Time[sec]'
set xrange [0:4]
set yrange [0:400]
set xtics (0,1,2,3,4)
set ytics (0,100,200,300,400)
set key left
plot "basic_problemSizeVsTime.txt" using 1:2 with linespoints title 'Serial Code', \
"basic_problemSizeVsTime.txt" using 1:3 with linespoints title 'Inner Loop Parallel', \
"basic_problemSizeVsTime.txt" using 1:4 with linespoints title 'Outer Loop Parallel', \

set output 'basic_problemSizeVsSpeedup.tex'
set title 'Basic Binary classification: Variation in speedup compared to the number of training samples iterations: 1000'
set xlabel 'Training Samples[log base 10]'
set ylabel 'Speedup'
set xrange [0:4]
set yrange [0:14]
set xtics (0,1,2,3,4)
set ytics (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14)
set key left
plot "basic_problemSizeVsSpeedup.txt" using 1:2 with linespoints title 'Speedup: Inner parallel', \
"basic_problemSizeVsSpeedup.txt" using 1:3 with linespoints title 'Speedup: Outer parallel', \

set output 'basic_coresVsSpeedup.tex'
set title 'Basic Binary classification: Variation in speedup compared to the number of threads'
set xlabel 'No. of Cores'
set ylabel 'Speedup'
set xrange [0:14]
set yrange [0:14]
set xtics (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14)
set ytics (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14)
set key left
plot "basic_coresVsSpeedup_size100.txt" using 1:2 with linespoints title 'Samples: 100, Inner parallel', \
"basic_coresVsSpeedup_size100.txt" using 1:3 with linespoints title 'Samples: 100, Inner parallel', \
"basic_coresVsSpeedup_size1499.txt" using 1:2 with linespoints title 'Samples: 1499, Outer parallel', \
"basic_coresVsSpeedup_size1499.txt" using 1:3 with linespoints title 'Samples: 1499, Outer parallel', \

set terminal epslatex size 5,3 color colortext

set output 'iterationVsSpeedup.tex'
set title 'Variation in speedup compared to the number of training samples in log scale samples: 60000'
set xlabel 'Training Samples[log base 10]'
set ylabel 'Speedup'
set xrange [0:4]
set yrange [0:4]
set xtics (0,1,2,3,4)
set ytics (0,1,2,3,4)
set key left
plot "iterationVsSpeedup.txt" using 1:2 with linespoints title 'Speedup'

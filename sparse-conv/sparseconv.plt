set terminal postscript "Times, 20" eps color dashed size 3,1.8
set loadpath '~/gnuplot-palettes'
load "blues.pal"
set output "sparseconv.ps"

set xlabel "Square root of Cin * Cout" font "Times, 20"

unset key
set datafile missing '-'
set xtics border in scale 0,0 nomirror norotate left autojustify font "Times, 16"
set ytics border in scale 1,0.5 nomirror norotate left autojustify
set key top font "Helvetica, 18"
set key autotitle columnhead

set yrange [0.5: 3]
set xrange [30: 320]
set logscale y 2
set logscale x 2
set border 15
set origin 0, 0
set size 1, 1

set ylabel "Normalized Speedup(x)" font "Times, 24"
set xtics border in scale 0,0 nomirror norotate left autojustify


NO_ANIMATION = 1
plot 'sparseconv.dat' using ($1):($2) w points pt 7 ps 1 lc "#2166AC", \
    '' u ($1):($3) with lp pt 0 lw 2 lc "#000000"

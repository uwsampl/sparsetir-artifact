from sparsetir_artifact import plot

prelude = """
set loadpath '~/gnuplot-palettes'
load "blues.pal"
set terminal postscript "Times, 20" eps color dashed size 8,2.5
set style fill  solid 1.00 border lt -1
set xlabel "Dataset" font "Times, 24" offset 20

set multiplot layout 1, 2
"""

text = """
set title "Speedup" font "Helvetica, 24" offset 0,-2.5
set rmargin 0
unset key
set border 7
set style histogram clustered gap 1 title textcolor lt -1
set style data histograms
set key top reverse maxrows 1 Left font "Helvetica,14" at 11, 200 width -4
set xtics border in scale 0,0 nomirror norotate left autojustify
set ytics border in scale 1,0.5 nomirror norotate left autojustify
set ylabel "Normalized Speedup(x)" font "Times, 24"
set yrange [0.03:100]
set logscale y 10 
set origin 0, 0
set offset -0.3, -0.3, 0, 0
set size 0.5, 1
to_str(x) = (x > 0) ? (sprintf("%.1f", x)): (x == -1 ? "N/A" : "OOM")
y_pos(x) = x * 1.1
y_val(x) = ((x > 0) ? x : 0.)
set arrow from -1,1 to 5,1 nohead dt 3 lw 3
set arrow from 5,0.03 to 5,100 nohead lw 3
"""


mem_text = """
unset arrow
set title "GPU Memory Footprint(GB)" font "Helvetica, 24"
set format y ""
set ylabel
set lmargin 0
unset border
set border 13
unset ytics
set xlabel " " font "Times,24"
set style histogram clustered gap 1 title textcolor lt -1
set y2tics border in scale 1,0.5 nomirror norotate right autojustify
unset key
set y2range [0.4:80000]
set logscale y2 10
set format y2 "10^%T"
set style data histograms
set origin 0.5, 0
set offset -0.3, -0.3, 0, 0
set size 0.45, 1
to_str(x) = (x > 0) ? (sprintf("%d", x)): (x == -1 ? "N/A" : "OOM")
y_pos(x) = x * 1.1
y_val(x) = ((x > 0) ? x : 0.)
"""
 

plot(
    "rgcn-e2e",
    prelude,
    [["rgcn-e2e", text], ["rgcn-e2e-mem", mem_text]],
    """with labels notitle rotate left font ",12" """,
    lambda x: -1.35 + 0.14 * x,
    [
        "p 3",
        "p 3",
        "p 3",
        "p 6",
        "p 1",
        "p 2",
    ],
    ["2", "5", "4", "6", "3", "0"],
    axes=[1, 2]
)

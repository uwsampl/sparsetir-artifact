from sparsetir_artifact import plot

prelude = """
set loadpath '~/gnuplot-palettes'
load "blues.pal"
set terminal postscript "Times, 20" eps color dashed size 2,4
set style fill  solid 1.00 border lt -1
#set xlabel "Dataset" font "Times, 24" offset 10 rotate by 90

set multiplot layout 1, 1
"""

text = """
set label "Dataset" rotate by 90 font "Times, 24" right at -1,0.4
unset key
set border 15
set style histogram clustered gap 1 title textcolor lt -1
set style data histograms
set xtics border in scale 0,0 nomirror rotate by 90 right #left autojustify
set ytics border in scale 1,0.5 nomirror rotate by 90 offset 0,0.6 #left autojustify
set ylabel "Normalized Speedup against DGL" offset 0,-1 font "Times, 24"
set yrange [0.5:2.4]
set origin 0, 0
set offset -0.3, -0.3, 0, 0
set size 1, 1
to_str(x) = (x > 0) ? ((x < 10) ? sprintf("%.2f", x) : sprintf("%d", x)): (x == -1 ? "N/A" : "OOM")
y_pos(x) = x + 0.1
y_val(x) = ((x > 0) ? x : 0.)
set arrow from -1,1 to 6.5,1 nohead lw 3 dt 3
"""

plot(
    "graphsage-e2e",
    prelude,
    [["graphsage-e2e", text]],
    """with labels notitle rotate by 90 font ",17" """,
    lambda x: -1.0 + 0.26 * x,
    [
        "p 2",
    ],
    ["0"],
)

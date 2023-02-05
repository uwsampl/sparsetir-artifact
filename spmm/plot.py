from sparsetir_artifact import plot

prelude = """
set loadpath '~/gnuplot-palettes'
load "blues.pal"
set terminal postscript "Times, 20" eps color dashed size 6,2.5
set style fill  solid 1.00 border lt -1
set xlabel "Dataset" font "Times, 24"
"""

text = """
set border 15
set style histogram clustered gap 1 title textcolor lt -1
set style data histograms
set key top maxrows 6 width 0 font "Helvetica,18" Left reverse at 6, 3.3
set xtics border in scale 0,0 nomirror norotate left autojustify font "Times, 18"
set ytics border in scale 1,0.5 nomirror norotate left autojustify
set ylabel "Normalized Speedup" font "Times, 24"
set yrange [0:3.5]
set origin 0, 0
set offset -0.3, -0.3, 0, 0
set size 1, 1
to_str(x) = (x > 0) ? ((x < 10) ? sprintf("%.1f", x) : sprintf("%d", x)): (x == -1 ? "N/A" : "OOM")
y_pos(x) = x + 0.1
y_val(x) = ((x > 0) ? x : 0.)
set arrow from -1,1 to 6.7,1 nohead dt 3 lw 3
"""


plot(
    "spmm",
    prelude,
    [["spmm", text]],
    """with labels notitle font ",12" """,
    lambda x: -1.37 + 0.143 * x,
    [
        "p 3",
        "p 3",
        "p 3",
        "p 6",
        "p 1",
        "p 2",
    ],
    ["2", "5", "4", "6", "3",  "0"],
)

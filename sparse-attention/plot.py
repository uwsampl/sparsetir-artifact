from sparsetir_artifact import plot

prelude = """
set loadpath '~/gnuplot-palettes'
load "blues.pal"
set terminal postscript "Times, 20" eps color dashed size 5,3
set style fill  solid 1.00 border lt -1
set xlabel "Sparse Pattern" font "Times, 28" offset 10, -1
set multiplot layout 1, 2
"""

spmm_text = """
set title "Multi-Head SpMM" font "Helvetica, 24" offset 0, -3
# set key top fixed maxrows 1 right Right horizontal font "Courier,18"
set key top maxrows 1 width -1 Right font "Helvetica, 18" at 5, 10
set rmargin 0
set border 7
set style histogram clustered gap 1 title textcolor lt -1
set style data histograms
set xtics axis in scale 0,0 nomirror rotate by 30 right font "Times, 18"
set ytics border in scale 1,0.5 nomirror norotate left autojustify
set ylabel "Normalized Speedup" font "Times, 28"
set yrange [0.05:25]
set logscale y 5
set origin 0, 0
set size 0.55, 1
to_str(x) = (x > 0) ? ((x < 10) ? sprintf("%.2f", x) : sprintf("%d", x)): (x == -1 ? "N/A" : "OOM")
y_pos(x) = x * 1.1
y_val(x) = ((x > 0) ? x : 0.)
"""

sddmm_text = """
set title "Multi-Head SDDMM" font "Helvetica, 24"
unset key
set lmargin 0
set format y ""
unset border
set border 13
unset ytics
unset ylabel
set xlabel " " font "Times,24"
set style histogram clustered gap 1 title textcolor lt -1
set style data histograms
set origin 0.55, 0
set size 0.45, 1
"""

plot(
    "blocksparse",
    prelude,
    [["bsr-spmm", spmm_text], ["bsr-sddmm", sddmm_text]],
    """with labels notitle rotate left font ",16" """,
    lambda x: -1.24 + 0.23 * x,
    [
        "p 3",
        "p 3",
        "p 2",
    ],
    ["2", "4", "0"],
)

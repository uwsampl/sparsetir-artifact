import subprocess
from typing import List, Tuple, Callable

plt_header = """
set loadpath './gnuplot-palettes'
load "greys.pal"
set output "{}.ps"
"""

p_list = [
    "p 1",
    "p 3",
    "p 3",
    "p 7",
    "p 3",
    "p 4",
    "p 2",
]

ls_list = [
    "0",
    "4",
    "1",
    "5",
    "6",
    "5",
    "6"
]

def plot(filename: str, prelude: str, subplots: List[Tuple[str, str]], label_str: str, label_x_offset_func: Callable):
    with open(filename + ".plt", 'w') as f_out:
        f_out.write(plt_header.format(filename) + "\n")
        f_out.write(prelude + "\n")
        for subplot in subplots:
            name, text = subplot
            f_out.write(text + "\n")
            with open(name + ".dat", "r") as f_in:
                lines = f_in.readlines()
                num_rows = len(lines)
                num_cols = len(lines[0].split())
            for i in range(num_cols - 1):
                fmt_str = "fs {} fc ls {} lw 3 ti col".format(p_list[i], ls_list[i])
                if i == 0:
                    f_out.write("""plot "{}" u (y_val($2)):xtic(1) {},\\\n""".format(name + ".dat", fmt_str))
                else:
                    f_out.write("""'' u (y_val(${})) {},\\\n""".format(i + 2, fmt_str))
                f_out.write("""'' u ($0+({})):(y_pos(${})):(to_str(${})) {}""".format(label_x_offset_func(i), i + 2, i + 2, label_str))
                if i != num_cols - 2:
                    f_out.write(",\\\n")
                else:
                    f_out.write("\n")

    subprocess.call(["gnuplot", filename + ".plt"])
    subprocess.call(["epstopdf", filename + ".ps"])
    subprocess.call(["rm", filename + ".plt"])
    subprocess.call(["rm", filename + ".ps"])

if __name__ == "__main__":
    prelude = """
set terminal postscript "Times, 20" eps color dashed size 10,3
set style fill   solid 1.00 border lt -1
set xlabel "Dataset" font "Times, 24" offset 30
set multiplot layout 1, 2
    """

    rgcn_text = """
set title "RGCN" font "Times, 24"
set rmargin 0
unset key
set border 3
set style histogram clustered gap 1 title textcolor lt -1
set style data histograms
set xtics border in scale 0,0 nomirror norotate left autojustify
set ytics border in scale 1,0.5 nomirror norotate left autojustify
set logscale y 10 
set ylabel "Time/ms" font "Times, 24"
set yrange [1:3000]
set origin 0, 0
set offset -0.3, -0.3, 0, 0
set size 0.55, 1
to_str(x) = (x > 0) ? ((x < 10) ? sprintf("%.1f", x) : sprintf("%d", x)): (x == -1 ? "N/A" : "OOM")
y_pos(x) = ((x > 1) ? x * 1.1 : 1.1)
y_val(x) = ((x > 0) ? x : 0.)
    """

    hgt_text = """
set title "HGT" font "Times, 24"
set lmargin 0
set rmargin 0
set format y ""
unset border
set border 1
unset ytics
unset ylabel
set xlabel " " font "Times,24"
set style histogram clustered gap 1 title textcolor lt -1
#set key top fixed right Right horizontal font "Courier,18"
#set key width 100
set key top vertical maxrows 1 width 0 font "Courier,18" Right at 3, 3000
#set key at screen 1, graph 1
set style data histograms
set origin 0.55, 0
set offset -0.3, -0.3, 0, 0
set size 0.45, 1
    """
    plot(
        "speed-heterograph",
        prelude,
        [["speed-rgcn", rgcn_text], ["speed-hgt", hgt_text]],
        """with labels notitle rotate left font ",15" """,
        lambda x: -1.38 + 0.13 * x
    )
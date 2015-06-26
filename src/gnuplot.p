set term pdfcairo
set output "loss.pdf"
set ylabel "loss"
set xlabel "timestep"
set format x "%.0s*10^%T"
plot '< awk -vn=100 -f average.awk losses.txt' using 1:2 notitle with line
plot '< awk -vn=1000 -f average.awk losses.txt' using 1:2 notitle with line


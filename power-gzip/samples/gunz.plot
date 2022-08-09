##
set title noenhanced "p9 decompress Z_files.tar"
!grep DECOMP runs2.txt > tmp2.log
!grep -A 1 gunzip runsgz.txt | grep -v gunzip | grep -v "\-\-" > tmp3.log
set term png
#
set key left top reverse Left
set logscale x 2
set format x "%L"
set xlabel "Output size, log_2(bytes)"
#
set format y "%g"
set ylabel "Throughput GB/s"
set output "decompress.png"
#
## plot 'tmp.log' using 4:(($4/($9/$6))/1.0e9) with linespoints title 'p9 h/w decompress b/w',
plot 'tmp2.log' using 4:(($4/($9/($6+$14)))/1.0e9) with linespoints title 'p9 decompress + page touch',\
     'tmp3.log' using 7:(($7/($1-$5+0.001))/1.0e9) every ::12::100 with linespoints title 'gunzip'




     

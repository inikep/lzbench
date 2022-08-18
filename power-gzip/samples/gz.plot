#
set title noenhanced "p9 compress Z_files.tar"
!grep COMP runs.txt | grep -v DECOM > tmp.log
!grep -A 1 gzip  runsgz.txt | grep -v gzip | grep -v "\-\-" > tmp2.log
set term png
#
set key left top reverse Left
set logscale x 2
set format x "%L"
set xlabel "Input size, log_2(bytes)"
#
set format y "%g"
set ylabel "Throughput GB/s"
set output "compress.png"
#
#plot 'tmp.log' using 5:(($5/($12/$9))/1.0e9) with linespoints title 'p9 h/w compress b/w',
plot  'tmp.log' using 5:(($5/(($12+$18)/$9))/1.0e9) with linespoints title 'p9 compress + page touch', \
      'tmp2.log' using 7:(($7/($1-$5+0.001))/1.0e9) every ::12::100 with linespoints title 'gzip'


     

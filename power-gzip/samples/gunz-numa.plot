##
set title noenhanced "P9 4 Threads 2 Engines Decompress: Total Throughput"
###
### in the multi thread case, just pick the first threads result assuming
### the remaining threads are same similar
### then multiply the result by the number of threads for aggregate thruput
###
!grep DECOMP runnuma1a.txt | grep "1\.head" > tmp1.log
###!grep DECOMP runnuma2.txt > tmp2.log
###!grep DECOMP runnuma3.txt > tmp3.log
###!grep -A 1 gunzip runsgz.txt | grep -v gunzip | grep -v "\-\-" > tmp3.log
set term png
#
set key left top reverse Left
set logscale x 2
set format x "%L"
set xlabel "Output size, log_2(bytes)"
#
set format y "%g"
###set format y "%.2s x 10^%S"
set ylabel "Throughput GByte/s"
set output "decompress.png"
#
## plot 'tmp.log' using 4:(($4/($9/$6))/1.0e9) with linespoints title 'p9 h/w decompress b/w',
##plot 'tmp2.log' using 4:(($4/($9/($6+$14)))/1.0e9) with linespoints title 'p9 decompress + page touch',\
##     'tmp3.log' using 7:(($7/($1-$5+0.001))/1.0e9) every ::12::100 with linespoints title 'gunzip'

##plot 'tmp1.log' using 4:(($4/($9/($6+$14)))/1.0e9) with linespoints title '4 threads 2 gzip engines'

# 4 * is the sum of 4 threads
plot 'tmp1.log' using ($4/$23):($16*4.0/1.0e9) with linespoints title '4 threads 2 gzip engines'
###     'tmp2.log' using ($4/$23):($16*4.0) with linespoints title '4 threads 2 gzip engines'
###     'tmp3.log' using ($4/$23):($16*8.0) with linespoints title '8 threads 2 gzip engines'




     

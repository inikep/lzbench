set terminal pngcairo size 1280,900 enhanced font 'Verdana,14'
set output 'compress.png'
set logscale x 2 ; set format x '2^{%L}';
set yrange [0:8]
set ylabel 'Total compress throughput (GB/s)'
set xlabel 'Source data size (bytes)'
set key center right
plot '<(grep "Total compress" nx-log.log | grep "threads 1,")' using 7:5 title '1 threads, NX' with linespoints,\
     '<(grep "Total compress" nx-log.log | grep "threads 2,")' using 7:5 title '2 threads, NX' with linespoints,\
     '<(grep "Total compress" nx-log.log | grep "threads 4,")' using 7:5 title '4 threads, NX' with linespoints,\
     '<(grep "Total compress" nx-log.log | grep "threads 8,")' using 7:5 title '8 threads, NX' with linespoints,\
     '<(grep "Total compress" nx-log.log | grep "threads 16,")' using 7:5 title '16 threads, NX' with linespoints,\
     '<(grep "Total compress" nx-log.log | grep "threads 32,")' using 7:5 title '32 threads, NX' with linespoints,\
     '<(grep "Total compress" nx-log.log | grep "threads 64,")' using 7:5 title '64 threads, NX' with linespoints,\
     '<(grep "Total compress" nx-log.log | grep "threads 80,")' using 7:5 title '80 threads, NX' with linespoints,\
     '<(grep "Total compress" zlib-log.log | grep "threads 1,")' using 7:5 title '1 threads, ZLIB' with lines,\
     '<(grep "Total compress" zlib-log.log | grep "threads 2,")' using 7:5 title '2 threads, ZLIB' with lines,\
     '<(grep "Total compress" zlib-log.log | grep "threads 4,")' using 7:5 title '4 threads, ZLIB' with lines,\
     '<(grep "Total compress" zlib-log.log | grep "threads 8,")' using 7:5 title '8 threads, ZLIB' with lines,\
     '<(grep "Total compress" zlib-log.log | grep "threads 16,")' using 7:5 title '16 threads, ZLIB' with lines,\
     '<(grep "Total compress" zlib-log.log | grep "threads 32,")' using 7:5 title '32 threads, ZLIB' with lines,\
     '<(grep "Total compress" zlib-log.log | grep "threads 64,")' using 7:5 title '64 threads, ZLIB' with lines,\
     '<(grep "Total compress" zlib-log.log | grep "threads 80,")' using 7:5 title '80 threads, ZLIB' with lines     


set terminal pngcairo size 640,480 enhanced font 'Verdana,14'
set output 'uncompress.png'
set logscale x 2 ; set format x '2^{%L}';
set ylabel 'Total uncompress throughput (GB/s)'
set yrange [0:11]
set xlabel 'Uncompressed data size (bytes)'
set key center right
plot '<(grep "Total uncompress" nx-log.log | grep "threads 1,")' using 7:5 title '1 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 2,")' using 7:5 title '2 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 4,")' using 7:5 title '4 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 8,")' using 7:5 title '8 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 16,")' using 7:5 title '16 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 32,")' using 7:5 title '32 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 64,")' using 7:5 title '64 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 80,")' using 7:5 title '80 threads, NX' with linespoints

set output 'uncompress12.png'
set logscale x 2 ; set format x '2^{%L}';
set ylabel 'Total uncompress throughput (GB/s)'
set yrange [0:11]
set xlabel 'Uncompressed data size (bytes)'
set key center right
plot '<(grep "Total uncompress" nx-log.log | grep "threads 1,")' using 7:5 title '1 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 2,")' using 7:5 title '2 threads, NX' with linespoints,\
     '<(grep "Total uncompress" zlib-log.log | grep "threads 1,")' using 7:5 title '1 threads, ZLIB' with lines,\
     '<(grep "Total uncompress" zlib-log.log | grep "threads 2,")' using 7:5 title '2 threads, ZLIB' with lines

set output 'uncompress48.png'
set logscale x 2 ; set format x '2^{%L}';
set ylabel 'Total uncompress throughput (GB/s)'
set yrange [0:11]
set xlabel 'Uncompressed data size (bytes)'
set key center right
plot '<(grep "Total uncompress" nx-log.log | grep "threads 4,")' using 7:5 title '4 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 8,")' using 7:5 title '8 threads, NX' with linespoints,\
     '<(grep "Total uncompress" zlib-log.log | grep "threads 4,")' using 7:5 title '4 threads, ZLIB' with lines,\
     '<(grep "Total uncompress" zlib-log.log | grep "threads 8,")' using 7:5 title '8 threads, ZLIB' with lines

set output 'uncompress1632.png'
set logscale x 2 ; set format x '2^{%L}';
set ylabel 'Total uncompress throughput (GB/s)'
set yrange [0:11]
set xlabel 'Uncompressed data size (bytes)'
set key center right
plot '<(grep "Total uncompress" nx-log.log | grep "threads 16,")' using 7:5 title '16 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 32,")' using 7:5 title '32 threads, NX' with linespoints,\
     '<(grep "Total uncompress" zlib-log.log | grep "threads 16,")' using 7:5 title '16 threads, ZLIB' with lines,\
     '<(grep "Total uncompress" zlib-log.log | grep "threads 32,")' using 7:5 title '32 threads, ZLIB' with lines

set output 'uncompress6480.png'
set logscale x 2 ; set format x '2^{%L}';
set ylabel 'Total uncompress throughput (GB/s)'
set yrange [0:11]
set xlabel 'Uncompressed data size (bytes)'
set key center right
plot '<(grep "Total uncompress" nx-log.log | grep "threads 64,")' using 7:5 title '64 threads, NX' with linespoints,\
     '<(grep "Total uncompress" nx-log.log | grep "threads 80,")' using 7:5 title '80 threads, NX' with linespoints,\
     '<(grep "Total uncompress" zlib-log.log | grep "threads 64,")' using 7:5 title '64 threads, ZLIB' with lines,\
     '<(grep "Total uncompress" zlib-log.log | grep "threads 80,")' using 7:5 title '80 threads, ZLIB' with lines









#!/bin/bash

plotfn=nx-log.log
TS=$(date +"%F-%H-%M-%S")
run_series_report=run_series_report_${TS}.csv

if [[ $# < 1 ]]; then
    echo "$0 needs a file as argument."
    exit 1
fi

if [[ $2 == "unsafe" ]]; then
    unsafe=true
else
    unsafe=false
fi

node=`numastat | head -1 | sed -e "s/ //g" | cut -c5`
for th in 1 4 16 32 64 80
do
    for a in `seq 0 2 20`  # size
    do
	b=$((1 << $a))
	nbyte=$(($b * 1024))
	rpt=$((1000 * 1000 * 1000 * 10)) # 10GB
	rpt=$(( ($rpt+$nbyte-1)/$nbyte )) # iters
	rpt=$(( ($rpt+$th-1)/$th )) # per thread
	rm -f junk2
	head -c $nbyte $1 > junk2;
	ls -l junk2;
	numactl -N $node ./compdecomp_th junk2 $th $rpt
    done
done  > $plotfn 2>&1

echo "comdecom,thread#,data size,bandwidth(GB/s)" > ${run_series_report}
for i in 1 4 16 32 64 80; do
    grep "Total compress" $plotfn | grep "threads $i," | awk '{print "compress,"$11 $7 $5 }' >> ${run_series_report}
done
for i in 1 4 16 32 64 80; do
    grep "Total uncompress" $plotfn | grep "threads $i," | awk '{print "uncompress,"$11 $7 $5 }' >> ${run_series_report}
done

# The following tests may crash the system on a failure.
if $unsafe; then
    head -c 1048576 $1 > junk2; # 1 Mb
    # Stress test for a system checkstop on kernel.
    ./compdecomp_th junk2 400 24
    # Stress test for disabling IRQ on kernel.
    echo "Testing for kernel disabling IRQ..."
    NX_GZIP_TIMEOUT_PGFAULTS=3 ./bad_irq_check junk2 400 96
    echo "Success!"
    # Stress test to check if the system handles many page faults.
    echo "Checking many random page faults..."
    ./rand_pfault_check junk2 100 480
    echo "Success!"
fi

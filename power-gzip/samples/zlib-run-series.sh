#!/bin/bash

plotfn=log.log

node=`numastat | head -1 | sed -e "s/ //g" | cut -c5`
for th in 1 2 4 8 16 32 64 80
do
    for a in `seq 0 20`  # size
    do
	b=$((1 << $a))
	nbyte=$(($b * 1024))
	rpt=$((1000 * 1000 * 1000 * 1)) # 1GB
	rpt=$(( ($rpt+$nbyte-1)/$nbyte )) # iters
	rpt=$(( ($rpt+$th-1)/$th )) # per thread
	rm -f junk2
	head -c $nbyte $1 > junk2;
	ls -l junk2;
	numactl -N $node ./compdecomp_th junk2 $th $rpt
    done
done  > $plotfn 2>&1



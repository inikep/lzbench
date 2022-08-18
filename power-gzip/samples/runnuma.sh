#!/bin/bash

#filename and number of instances
echo $1 $2

C=$2

rm -f *.nx.gunzip

for S in `seq 1 $C` 
do
    FN=$S.$1
	echo $FN
	rm -f $FN

	# create the unique file
    cp $1 $FN

	# FS cache warmup
    cat $FN > /dev/null
done

nodes=`numastat | head -1`
for S in `seq 1 $C`
do
	FN=$S.$1    
	for i in $nodes
	do
		node=`echo $i | sed -e "s/node//g"`
		if [ "$node" -lt "16" ]; then
			numactl -N $node ./gunzip_nx_test $FN  &
		fi
	done
done

wait


 

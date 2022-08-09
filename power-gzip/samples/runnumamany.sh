#!/bin/bash

echo $1

for SZ in 512 1K 2K 4K 8K 16K 32K 64K 128K
# 256K 512K 1M
do
	FN=head.$1.$SZ
	head -c $SZ $1 > $FN
	./gzip_nxfht_test $FN
        ./runnuma.sh $FN.nx.gz 4 		
	rm -f $FN $FN.nx.gz $FN.*.nx.gunzip
	echo
	sleep 2
done
 

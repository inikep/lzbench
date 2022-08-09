#!/bin/bash

echo $1

for SZ in 512 1K 2K 4K 8K 16K 32K 64K 128K 256K 512K 1M 2M 4M 8M 16M 32M 64M 128M 256M 512M 1G
do
	FN=head.$1.$SZ
	head -c $SZ $1 > $FN
	cat $FN > /dev/null
	/usr/bin/time  -f "\t%e real \t%U user \t%S sys" ./gzip_nxfht_test $FN
	echo	
	cat $FN.nx.gz > /dev/null
	/usr/bin/time  -f "\t%e real \t%U user \t%S sys" ./gunzip_nx_test $FN.nx.gz	
	rm -f $FN $FN.nx.gz $FN.*.nx.gunzip
	echo
done
 

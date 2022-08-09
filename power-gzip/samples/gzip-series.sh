#!/bin/bash -x

# sudo mkdir /mnt/ramdisk
# sudo mount -t tmpfs -o size=250G tmpfs /mnt/ramdisk/
#
# size the ramdisk according to the largest file you will have

OPATH=/mnt/ramdisk
rm -f $OPATH/*

IFN=$1 #supply an input file

# warmup file system cache
cat $IFN > /dev/null

node=`numastat | head -1 | sed -e "s/ //g" | cut -c5`
echo running nx-gzip
OFN=$OPATH/$(basename $IFN).nx.gz
time numactl -N $node ./zpipe < $IFN > $OFN
ls -l $IFN $OFN
echo
echo

# uncomment these if comparing to gzip
#
#echo running gzip
#for a in 1 2 3 4 5 6 7 8 9
#do
#    OFN=$OPATH/$(basename $IFN).$a.gz
#    rm -f $OPATH/*
#    time numactl -N $node gzip -$a $IFN -c > $OFN
#    ls -l $IFN $OFN
#    echo
#    echo
#done

rm -f $OPATH/*

# create a file to decompress; we could use gzip too
# but it takes too long
OFN=$OPATH/source.gz
./gzm < $IFN > $OFN

echo running nx uncompress
time numactl -N $node ./gzm -d < $OFN > /dev/null
ls -l $IFN $OFN
echo
echo

echo running gunzip
time numactl -N $node gunzip $OFN -c > /dev/null
ls -l $IFN $OFN
echo


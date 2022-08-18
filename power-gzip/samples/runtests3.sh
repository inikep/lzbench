#!/bin/bash

# compile gzm
# create a ramdisk as a working directory RUNDIR, for example
# mount -t tmpfs -o size=10G tmpfs /mnt/ramdisk/
# put uncompressible files in SOURCEDIR which is a subdirectory of where gzm resides
# you can use bzip2 or gzip -9 to make uncompressible files.
# this script will produce test files of various sizes.
# you can run multiple copies of this script.  each copy will be suffixed with
# the process ID.  If errors.* file is produced we have a problem.
# It may take hours to run the script. Faster times obtained by reducing
# the iterations from 200 to a smaller.

RUNDIR=/mnt/ramdisk
SOURCEDIR=./uncompressible
PID=$$

FIN=$RUNDIR/datain.$PID
FOUT=$RUNDIR/dataout.$PID
FGZ=$RUNDIR/data.z.$PID
FLOG=$RUNDIR/log.$PID
FERR=$RUNDIR/errors.$PID

for FSZ1 in `seq 10 30`
do
    for n in `seq 1 1000`
    do
	FSZ=$(((`od -A n -t u4 -N 4 /dev/urandom`) % (1<<$FSZ1) + 1024))
	#echo "file size " $FSZ >> $FLOG

	for fname in `ls $SOURCEDIR`
	do

	for ZGZ in 0 1 #zlib and gzip formats
	do
	    FSRC=$SOURCEDIR/$fname
	    echo "file name " $FSRC >> $FLOG

	    echo "file size" $FSZ >> $FLOG

	    rm -f $FIN $FOUT $FGZ

	    # make data
	    head -c $FSZ $FSRC > $FIN
	    rc=$?
	    if [[ $rc != 0 ]]; then
		echo "EEEEEE makedata bad return code: filename size:" $fname, $FSZ >> $FERR
	    fi

	    # compress
	    ./gzm -t $ZGZ < $FIN > $FGZ  2>> $FLOG
	    rc=$?
	    if [[ $rc != 0 ]]; then
		echo "EEEEEE gzm bad return code: filename size seed:" $fname, $FSZ >> $FERR
	    fi

	    # uncompress
	    ./gzm -d -t $ZGZ < $FGZ > $FOUT 2>> $FLOG
	    rc=$?
	    if [[ $rc != 0 ]]; then
		echo "EEEEEE gzm -d bad return code: filename size:" $fname, $FSZ >> $FERR
	    fi

	    echo "file" $fname "target size" $FSZ >> $FLOG
	    ls -l $FIN $FOUT >> $FLOG

	    # brute force comparison of the input and output files
	    cksum1=`cat $FIN | sha1sum`
	    cksum2=`cat $FOUT | sha1sum`
	    echo $cksum1 >> $FLOG
	    echo $cksum2 >> $FLOG

	    # if mismatch write the details in errors.$PID file
	    if [[ $cksum1 != $cksum2 ]]; then
		echo "EEEEEE checksum mismatch" >> $FERR
		echo "file" $fname "target size" $FSZ >> $FERR
		ls -l $FIN $FOUT >> $FERR
		echo $cksum1 >> $FERR
		echo $cksum2 >> $FERR
		echo >> $FERR
	    fi

	    echo "AAAAAA" >> $FLOG
	    echo "AAAAAA" >> $FLOG
	done
	done
    done
done

#!/bin/bash

# compile zpipe and makedata
# create a ramdisk as a working directory RUNDIR, for example
# mount -t tmpfs -o size=10G tmpfs /mnt/ramdisk/
# put seed files in SOURCEDIR which is a subdirectory of where zpipe and makedata resides.
# seed files are sample sources that makedata converts in to other files which
# may be compressed and decompressed for testing purposes.
# this script will produce test files 256 byte to 1GB in size. Large files
# take longer to process because of the sha1sum verification.
# for each size 200 different files will be produced (see the for loops below).
# you can run multiple copies of this script.  each copy will be suffixed with
# the process ID.  If errors.* file is produced we have a problem.
# It may take hours to run the script. Faster times obtained by reducing
# the iterations from 200 to a smaller.

RUNDIR=/mnt/ramdisk
SOURCEDIR=./test
PID=$$

FIN=$RUNDIR/datain.$PID
FOUT=$RUNDIR/dataout.$PID
FGZ=$RUNDIR/data.z.$PID
FLOG=$RUNDIR/log.$PID
FERR=$RUNDIR/errors.$PID

for FSZ in `seq 8 30`
do
    echo "file size " $FSZ >> $FLOG

    for n in `seq 1 200`
    do

	for fname in `ls $SOURCEDIR`
	do
	    FSRC=$SOURCEDIR/$fname
	    echo "file name " $FSRC >> $FLOG

	    seed=`od -A n -t u4 -N 4 /dev/urandom`
	    echo "seed" $seed >> $FLOG

	    rm -f $FIN $FOUT $FGZ

	    # make synthetic data
	    ./makedata -s $seed -b $FSZ < $FSRC > $FIN  2> /dev/null

	    rc=$?
	    if [[ $rc != 0 ]]; then
		echo "EEEEEE makedata bad return code: filename size seed:" $fname, $FSZ, $seed >> $FERR
	    fi

	    # compress
	    ./zpipe < $FIN > $FGZ  2>> $FLOG
	    rc=$?
	    if [[ $rc != 0 ]]; then
		echo "EEEEEE zpipe bad return code: filename size seed:" $fname, $FSZ, $seed >> $FERR
	    fi

	    # uncompress
	    ./zpipe -d < $FGZ > $FOUT 2>> $FLOG
	    rc=$?
	    if [[ $rc != 0 ]]; then
		echo "EEEEEE zpipe -d bad return code: filename size seed:" $fname, $FSZ, $seed >> $FERR
	    fi

	    echo "file" $fname "target size" $FSZ "seed" $seed >> $FLOG
	    ls -l $FIN $FOUT >> $FLOG

	    # brute force comparison of the input and output files
	    cksum1=`cat $FIN | sha1sum`
	    cksum2=`cat $FOUT | sha1sum`
	    echo $cksum1 >> $FLOG
	    echo $cksum2 >> $FLOG

	    # if mismatch write the details in errors.$PID file
	    if [[ $cksum1 != $cksum2 ]]; then
		echo "EEEEEE checksum mismatch" >> $FERR
		echo "file" $fname "target size" $FSZ "seed" $seed >> $FERR
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

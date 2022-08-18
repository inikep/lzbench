#!/bin/bash

. config.sh

input=${1}.source

decompress()
{
	cat ${2} | ${1} -cd - > ${2}.2
	mv ${2}.2 ${2}
}



set -e
f=$(mktemp -p .)
${WGET} -q -L "$(${SED} -n 1p ${input})" -O ${f}
case "$(${SED} -n 2p ${input})" in
  "none") ;;
  "bzip2") decompress ${BZIP2} ${f};;
  "gzip")  decompress ${GZIP}  ${f};;
  "xz")    decompress ${XZ}    ${f};;
  *) exit 1;;
esac
case "$(${SED} -n 3p ${input} | ${AWK} '{print $1}')" in
  "sha256")
    echo "$(${SED} -n 3p ${input} | ${AWK} '{print $2}')  ${f}" \
      | ${SHA256SUM} -c --status -
    ;;
  *) exit 1;;
esac;
mv ${f} ${1}.uncompressed

#!/bin/bash
# Generate *.source file from a URL.
# e.g. ./generate-source.sh <URL> > example.source

. config.sh

url=${1}

decompress()
{
  cat ${2} | ${1} -cd - > ${2}.2
  mv ${2}.2 ${2}
}

type=""

set -e
f=$(mktemp)
${WGET} -q -L "${url}" -O ${f}

case "$(${FILE} -bi ${f})" in
  "application/gzip"*) type="gzip";;
  "application/x-bzip2"*) type="bzip2";;
  "application/x-xz"*) type="xz";;
  *) type="none";;
esac

case ${type} in
  "bzip2") decompress ${BZIP2} ${f};;
  "gzip")  decompress ${GZIP}  ${f};;
  "xz")    decompress ${XZ}    ${f};;
esac

checksum=$(${SHA256SUM} ${f} | awk '{print $1}')
rm ${f}

echo "${url}"
echo ${type}
echo "sha256 ${checksum}"

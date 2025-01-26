#!/bin/bash

if [ $# -lt 2 ]
  then
    echo "Usage:"
    echo "    $0 [algorithm (lz4/snappy/cascaded/gdeflate/bitcomp)] [directory]"
    exit 1
fi

ALGO="$1"
DIR="$2"

BINARY="../build/bin/benchmark_${ALGO}_chunked"

# Create a temp directory for all the logs
LOGDIR="$(mktemp -d)"
trap 'rm -rf -- "${LOGDIR}"' EXIT

output_header() {
  echo dataset, uncompressed bytes, compression ratio, "compression throughput (GB/s)", "decompression throughput (GB/s)"
  return 0
}

run_benchmark () {
  CMD="$1 -f $2"
  FILENAME="${LOGDIR}/$(basename $1)_$(basename $2).log"
  ${CMD} &> "${FILENAME}"
  bytes=$(awk '/^uncompressed /{print $3}' "${FILENAME}")
  ratio=$(awk '/compressed ratio:/{print $5}' "${FILENAME}")
  comp_throughput=$(awk '/^compression throughput /{print $4}' "${FILENAME}")
  decomp_throughput=$(awk '/^decompression throughput /{print $4}' "${FILENAME}")
  echo $(basename $2), $bytes, $ratio, $comp_throughput, $decomp_throughput
  return 0
}

# Run the benchmark for all files in DIR
output_header
for fname in ${DIR}/*
do
  if [[ -f "${fname}" ]]
  then
    run_benchmark "$BINARY" "$fname"
  fi
done

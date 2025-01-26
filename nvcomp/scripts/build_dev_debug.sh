#!/bin/bash

BUILD="./build"

if [[ ! -d "${BUILD}" ]]; then
  mkdir "${BUILD}"
fi

echo "=============="
echo "Building Debug"
echo "=============="

RV=0

pushd "${BUILD}"

cmake .. \
  -DDEVEL=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_BENCHMARKS=ON \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug && \
  make -j && \
  make test
RV=$?

for bench in bin/benchmark_*_synth; do
  if [[ "${RV}" == "0" ]]; then
    echo "Running ${bench}..."
    "${bench}" 
    RV=$?
  fi
done

popd

if [[ "${RV}" == "0" ]]; then
  echo "==============="
  echo "Build Succeeded"
  echo "==============="
else
  echo "============"
  echo "Build Failed"
  echo "============"
fi
exit ${RV} 

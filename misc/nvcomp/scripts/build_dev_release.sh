#!/bin/bash

BUILD="./build"

if [[ ! -d "${BUILD}" ]]; then
  mkdir "${BUILD}"
fi

pushd "${BUILD}"

echo "================"
echo "Building Release"
echo "================"

cmake .. \
  -DDEVEL=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_BENCHMARKS=ON \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release && \
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

#!/user/bin/python
##
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# # Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# # Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# # Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import sys
import csv
import os
import subprocess

if len(sys.argv) != 3:
	print "Usage: python allgather_runall.py <binary input filename> <Max number of GPUs>"
	sys.exit(0)

filename = sys.argv[1]
maxgpus = sys.argv[2]
print "Starting benchmark on file:", filename, " - using 2 -", maxgpus, "GPUs"

log = open('allgather-results.log', 'w')

with open('allgather-results.csv', 'w') as f:
	thewriter = csv.writer(f)
	thewriter.writerow(['Filename', 'num GPUs', 'chunks per GPU', 'No-comp throughput', 'LZ4 throughput', 'Cascaded throughput'])

	for gpus in range(2, int(maxgpus)+1):
		print "Testing using", gpus, "GPUs..."
		for chunks in [1,2,4]:
			cmd = './bin/benchmark_allgather -f ' + str(filename) + ' -g ' + str(gpus) + ' -h ' + str(gpus*chunks) + ' -c none'
			log.write(cmd + "\n")
			result = subprocess.check_output(cmd, shell=True)
			log.write(result + "\n")
                        nocomp = result.split()[-1]

			cmd = './bin/benchmark_allgather -f ' + str(filename) + ' -g ' + str(gpus) + ' -h ' + str(gpus*chunks) + ' -c lz4'
			log.write(cmd + "\n")
			result = subprocess.check_output(cmd, shell=True)
			log.write(result + "\n")
			lz4 = result.split()[-1]

			cmd = './bin/benchmark_allgather -f ' + str(filename) + ' -g ' + str(gpus) + ' -h ' + str(gpus*chunks) + ' -c cascaded'
			log.write(cmd + "\n")
			result = subprocess.check_output(cmd, shell=True)
			log.write(result + "\n")
			cascaded = result.split()[-1]
			thewriter.writerow([filename, gpus, chunks, nocomp,lz4,cascaded])

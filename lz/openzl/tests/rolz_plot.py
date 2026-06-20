#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.


import argparse
import csv
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output_directory")

args = parser.parse_args()

data_by_file = {}
data_size_by_file = {}

with open(args.input, "r") as csv_input:
    csv_reader = csv.DictReader(csv_input)
    for row in csv_reader:
        filename = row["filename"]
        algorithm = row["algorithm"]
        parameters = row["parameters"]
        data_size = int(row["data_size"])

        data_size_by_file[filename] = data_size

        if algorithm == "zstd" and int(parameters) < 0:
            continue

        data_by_algorithm = data_by_file.setdefault(filename, {})
        data_by_parameters = data_by_algorithm.setdefault(algorithm, {})
        data = data_by_parameters.setdefault(parameters, {})
        data["compression_ratio"] = data_size / int(row["compressed_size"])
        data["compression_mbps"] = data_size / int(row["compression_nanos"]) * 1000
        data["decompression_mbps"] = data_size / int(row["decompression_nanos"]) * 1000

for filename, data_by_algorithm in data_by_file.items():
    basename = os.path.basename(filename)
    data_size = data_size_by_file[filename]
    title = f"{basename} ({data_size} B)"
    output_prefix = os.path.join(args.output_directory, basename)
    output_data = output_prefix + ".dat"
    output_plot = output_prefix + ".png"
    gnuplot_args = [
        "gnuplot",
        "-e",
        f"set terminal png; set output '{output_plot}'; set title '{title}'; set xlabel 'Compression MB/s'; set ylabel 'Compression Ratio'; set autoscale; set logscale x 2; plot for [IDX=0:1] '{output_data}' i IDX u 1:2 w points title columnheader(1)",
    ]
    gnuplot_data = ""
    for algorithm, data_by_parameters in data_by_algorithm.items():
        gnuplot_data += f"{algorithm}\n"
        for data in data_by_parameters.values():
            compression_mbps = data["compression_mbps"]
            compression_ratio = data["compression_ratio"]
            gnuplot_data += f"{compression_mbps} {compression_ratio}\n"
        gnuplot_data += "\n\n"
    gnuplot_data = gnuplot_data.strip() + "\n"

    with open(output_data, "w") as f:
        f.write(gnuplot_data)

    gnuplot = subprocess.Popen(
        gnuplot_args,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    (stdout, stderr) = gnuplot.communicate(gnuplot_data.encode("ascii"))
    if gnuplot.returncode != 0:
        print(filename)
        print(stderr)
        print(stdout)

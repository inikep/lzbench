// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

const std::string kResultsFile = "results.csv";

struct AggregateStats {
    size_t origSize;
    size_t compressedSize;
    double ctimeMs;
    double dtimeMs;
};

static void processDir(const std::filesystem::path& dirPath)
{
    std::vector<std::filesystem::path> files;
    if (std::filesystem::exists((dirPath / "high"))) {
        assert(std::filesystem::exists((dirPath / "low")));
        files.push_back(dirPath / "high" / kResultsFile);
        files.push_back(dirPath / "low" / kResultsFile);
    } else {
        files.push_back(dirPath / kResultsFile);
    }
    AggregateStats stats{
        .origSize       = 0,
        .compressedSize = 0,
        .ctimeMs        = 0,
        .dtimeMs        = 0,
    };
    for (const auto& file : files) {
        std::ifstream in(file);
        std::string line;
        std::getline(in, line); // skip header
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            std::vector<std::string> fields;
            std::stringstream ss(line);
            std::string field;
            while (std::getline(ss, field, ',')) {
                fields.push_back(field);
            }
            assert(fields.size() == 6);
            stats.origSize += std::stoull(fields[0]);
            stats.compressedSize += std::stoull(fields[1]);
            stats.ctimeMs += std::stod(fields[3]);
            stats.dtimeMs += std::stod(fields[4]);
        }
    }

    std::cout << "OpenZL," << dirPath << ',';
    std::cout << (double)stats.origSize / (double)stats.compressedSize << ',';
    std::cout << (double)stats.origSize / 1024. / 1024. / (double)stats.ctimeMs
                    * 1000.
              << ',';
    std::cout << (double)stats.origSize / 1024. / 1024. / (double)stats.dtimeMs
                    * 1000.
              << std::endl;
}

int main(int argc, char** argv)
{
    std::string dirName = argv[1];
    std::cout << "Analyzing " << dirName << std::endl;
    std::cout
            << "Compressor Name,Dataset,Compression Ratio,Compression Speed MiBps,Decompression Speed MiBps\n";
    std::filesystem::path dirPath(dirName);
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.is_directory()) {
            processDir(entry.path());
        }
    }
    return 0;
}

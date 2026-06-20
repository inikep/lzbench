// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "Logger.h"
#include <cstdlib>

namespace openzl::tools::logger {

bool Logger::shouldLog(LogLevel level)
{
    return static_cast<int>(level) <= instance().global_verbosity;
}

void Logger::clearLine()
{
    fprintf(stderr, "\r");
    for (int i = 0; i < PADDING_SIZE; ++i) {
        fprintf(stderr, " ");
    }
    fprintf(stderr, "\r");
    fflush(stderr);
}

void Logger::finalizeProgressIfActive()
{
    if (instance().progress_line_active) {
        // Clear the current line
        clearLine();
        // The log line will be printed after this function returns
        // Then we need to re-print the progress line
        // We don't set progress_line_active to false here since we want to
        // maintain the progress state
    }
}

void Logger::reprintProgressIfActive()
{
    if (instance().progress_line_active
        && shouldLog(instance().progress_level)) {
        // Re-print the stored progress message
        update(instance().progress_level,
               "%s",
               instance().progress_message.c_str());
    }
}

} // namespace openzl::tools::logger

#pragma once

#include <chrono>
#include <fstream>
#include <string>

namespace infinilm::agent_debug {

inline void log(const char *location, const char *message, const char *hypothesis_id,
                const std::string &data_json, const char *run_id = "pre-fix") {
    // #region agent log
    std::ofstream f("/workspace/.cursor/debug-8a7b5d.log", std::ios::app);
    if (!f) {
        return;
    }
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    f << "{\"sessionId\":\"8a7b5d\",\"runId\":\"" << run_id << "\",\"hypothesisId\":\""
      << hypothesis_id << "\",\"location\":\"" << location << "\",\"message\":\"" << message
      << "\",\"data\":" << data_json << ",\"timestamp\":" << ts << "}\n";
    // #endregion
}

} // namespace infinilm::agent_debug

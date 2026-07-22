#pragma once

#include <cstddef>
#include <string>

namespace infinilm::engine::distributed {

/**
 * Configuration for a one-shot TCP rendezvous among distributed processes.
 * Participant 0 is the coordinator. All other participants connect to it at
 * coordinator_addr:coordinator_port.
 */
struct TcpRendezvousConfig {
    std::string coordinator_addr;
    int coordinator_port;
    int participant_count;
    int participant_rank;
};

/**
 * Broadcast an opaque, fixed-size payload from participant 0.
 *
 * The call blocks until the coordinator has served every participant, or until
 * a participant has received the payload. Nonzero participants first register
 * their rank, allowing the coordinator to reject duplicates and invalid ranks.
 * The API does not interpret the payload and can bootstrap any distributed
 * feature, such as PP or EP. Every participant must pass the same payload_size.
 * This is a one-shot rendezvous; callers establish their long-lived transport
 * after this function returns.
 */
void broadcast_rendezvous_payload(const TcpRendezvousConfig &config,
                                  void *payload,
                                  size_t payload_size);

} // namespace infinilm::engine::distributed

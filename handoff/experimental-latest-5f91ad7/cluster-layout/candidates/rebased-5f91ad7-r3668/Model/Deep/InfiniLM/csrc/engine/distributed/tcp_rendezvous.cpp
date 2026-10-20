#include "tcp_rendezvous.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <netdb.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace infinilm::engine::distributed {
namespace {

class SocketFd {
public:
    explicit SocketFd(int fd = -1) : fd_(fd) {}
    ~SocketFd() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    SocketFd(const SocketFd &) = delete;
    SocketFd &operator=(const SocketFd &) = delete;
    SocketFd(SocketFd &&other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    SocketFd &operator=(SocketFd &&other) noexcept {
        if (this != &other) {
            reset(other.fd_);
            other.fd_ = -1;
        }
        return *this;
    }

    int get() const { return fd_; }
    void reset(int fd = -1) {
        if (fd_ >= 0) {
            close(fd_);
        }
        fd_ = fd;
    }

private:
    int fd_;
};

void throw_errno(const char *operation) {
    throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

void send_all(int fd, const void *data, size_t size) {
    const char *ptr = static_cast<const char *>(data);
    while (size > 0) {
        const ssize_t sent = send(fd, ptr, size, 0);
        if (sent <= 0) {
            throw_errno("TCP rendezvous send");
        }
        ptr += sent;
        size -= static_cast<size_t>(sent);
    }
}

void recv_all(int fd, void *data, size_t size) {
    char *ptr = static_cast<char *>(data);
    while (size > 0) {
        const ssize_t received = recv(fd, ptr, size, 0);
        if (received <= 0) {
            throw_errno("TCP rendezvous recv");
        }
        ptr += received;
        size -= static_cast<size_t>(received);
    }
}

SocketFd connect_to_coordinator(const TcpRendezvousConfig &config) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *addresses_raw = nullptr;
    const std::string port = std::to_string(config.coordinator_port);
    const int status = getaddrinfo(config.coordinator_addr.c_str(),
                                   port.c_str(),
                                   &hints,
                                   &addresses_raw);
    if (status != 0) {
        throw std::runtime_error("TCP rendezvous getaddrinfo: " + std::string(gai_strerror(status)));
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(addresses_raw,
                                                                 freeaddrinfo);

    for (int attempt = 0; attempt < 300; ++attempt) {
        for (addrinfo *address = addresses.get(); address != nullptr; address = address->ai_next) {
            SocketFd fd(socket(address->ai_family, address->ai_socktype, address->ai_protocol));
            if (fd.get() < 0) {
                continue;
            }
            if (connect(fd.get(), address->ai_addr, address->ai_addrlen) == 0) {
                return fd;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("failed to connect to TCP rendezvous coordinator");
}

} // namespace

void broadcast_rendezvous_payload(const TcpRendezvousConfig &config,
                                  void *payload,
                                  size_t payload_size) {
    if (config.participant_count < 1) {
        throw std::runtime_error("TCP rendezvous participant_count must be at least 1");
    }
    if (config.participant_rank < 0 || config.participant_rank >= config.participant_count) {
        throw std::runtime_error("TCP rendezvous participant_rank is out of range");
    }
    if (payload == nullptr && payload_size != 0) {
        throw std::runtime_error("TCP rendezvous payload must not be null");
    }
    if (config.participant_count == 1) {
        return;
    }

    if (config.participant_rank != 0) {
        auto fd = connect_to_coordinator(config);
        const uint32_t network_rank = htonl(static_cast<uint32_t>(config.participant_rank));
        send_all(fd.get(), &network_rank, sizeof(network_rank));
        recv_all(fd.get(), payload, payload_size);
        spdlog::info(
            "Distributed bootstrap connection established: role=participant, rank={}, coordinator={}:{}",
            config.participant_rank,
            config.coordinator_addr,
            config.coordinator_port);
        return;
    }

    SocketFd listen_fd(socket(AF_INET, SOCK_STREAM, 0));
    if (listen_fd.get() < 0) {
        throw_errno("TCP rendezvous socket");
    }
    int reuse_address = 1;
    setsockopt(listen_fd.get(),
               SOL_SOCKET,
               SO_REUSEADDR,
               &reuse_address,
               sizeof(reuse_address));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<uint16_t>(config.coordinator_port));
    if (bind(listen_fd.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        throw_errno("TCP rendezvous bind");
    }
    if (listen(listen_fd.get(), config.participant_count - 1) != 0) {
        throw_errno("TCP rendezvous listen");
    }

    spdlog::info(
        "Distributed bootstrap rendezvous listening: role=coordinator, endpoint=0.0.0.0:{}, participants={}",
        config.coordinator_port,
        config.participant_count);

    std::vector<bool> registered(config.participant_count, false);
    registered[0] = true;
    for (int connection_idx = 1; connection_idx < config.participant_count; ++connection_idx) {
        SocketFd client_fd(accept(listen_fd.get(), nullptr, nullptr));
        if (client_fd.get() < 0) {
            throw_errno("TCP rendezvous accept");
        }
        uint32_t network_rank = 0;
        recv_all(client_fd.get(), &network_rank, sizeof(network_rank));
        const int participant_rank = static_cast<int>(ntohl(network_rank));
        if (participant_rank <= 0 || participant_rank >= config.participant_count || registered[participant_rank]) {
            throw std::runtime_error("TCP rendezvous received an invalid or duplicate participant rank");
        }
        registered[participant_rank] = true;
        send_all(client_fd.get(), payload, payload_size);
        spdlog::info(
            "Distributed bootstrap connection established: role=coordinator, participant_rank={}, connected={}/{}",
            participant_rank,
            connection_idx,
            config.participant_count - 1);
    }
    spdlog::info(
        "Distributed bootstrap rendezvous complete: role=coordinator, participants={}",
        config.participant_count);
}

} // namespace infinilm::engine::distributed

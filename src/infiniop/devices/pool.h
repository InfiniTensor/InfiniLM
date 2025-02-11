#ifndef __POOL_H__
#define __POOL_H__

#include <atomic>
#include <mutex>
#include <optional>

template<class T>
class Pool {
public:
    Pool() : _head(nullptr) {}

    Pool(const Pool &) = delete;

    Pool(Pool &&pool) noexcept : _head(pool._head.exchange(nullptr)) {}

    ~Pool() {
        while (this->pop()) {}
    }

    void push(T &&val) const {
        Node<T> *new_node = new Node<T>(std::move(val));
        new_node->next = _head.load();
        while (!_head.compare_exchange_weak(new_node->next, new_node));
    }

    std::optional<T> pop() const {
        Node<T> *top = _head.load();
        Node<T> *new_head = nullptr;
        do {
            if (!top) {
                return std::nullopt;
            }
            new_head = top->next;
        } while (!_head.compare_exchange_weak(top, new_head));
        return {std::move(top->data)};
    }

private:
    template<class U>
    struct Node {
        U data;
        Node<U> *next;
        Node(U &&data) : data(data), next(nullptr) {}
    };

    mutable std::atomic<Node<T> *> _head;
};

#endif // __POOL_H__

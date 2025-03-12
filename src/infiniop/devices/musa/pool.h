#ifndef __POOL_MUSA_H__
#define __POOL_MUSA_H__

#include <atomic>
#include <mutex>
#include <optional>

template <class T>
class Pool {
public:
    Pool() : _head(nullptr) {}

    Pool(const Pool &) = delete;

    Pool(Pool &&pool) noexcept : _head(pool._head.exchange(nullptr)) {}

    ~Pool() {
        while (this->pop()) {}
    }

    void push(T *val) const {
        Node<T> *new_node = new Node<T>(val);
        new_node->next = _head.load();
        while (!_head.compare_exchange_weak(new_node->next, new_node)) {}
    }

    T *pop() const {
        Node<T> *top = _head.load();
        Node<T> *new_head = nullptr;
        do {
            if (!top) {
                return nullptr;
            }
            new_head = top->next;
        } while (!_head.compare_exchange_weak(top, new_head));
        return top->data;
    }

private:
    template <class U>
    struct Node {
        U *data;
        Node<U> *next;
        Node(U *data) : data(data), next(nullptr) {}
    };

    mutable std::atomic<Node<T> *> _head;
};

#endif // __POOL_MUSA_H__

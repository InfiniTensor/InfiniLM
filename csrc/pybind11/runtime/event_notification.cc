#include <pybind11/pybind11.h>

#include "runtime/event_transaction.hpp"

#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace py = pybind11;

namespace infinilm::runtime {
namespace {

// Python owns the stream objects; EventTransaction owns the Runtime. Native
// drain/destruction precedes release of the borrowed streams' Python owners.
class StreamNotification {
public:
    StreamNotification(Device device, std::uintptr_t producer,
                       std::uintptr_t consumer, unsigned int event_flags,
                       py::object producer_owner, py::object consumer_owner,
                       bool fixed_context)
        : owner_(std::this_thread::get_id()), fixed_context_(fixed_context),
          producer_owner_(std::move(producer_owner)),
          consumer_owner_(std::move(consumer_owner)) {
        if (producer_owner_.is_none() || consumer_owner_.is_none())
            throw std::invalid_argument("borrowed streams require live owner objects");
        py::gil_scoped_release release;
        auto runtime = Runtime::create(device);
        runtime->enableOwnerDrainFailStop();
        auto transaction = std::make_unique<EventTransaction>(
            runtime, reinterpret_cast<infini::rt::runtime::Stream>(producer),
            reinterpret_cast<infini::rt::runtime::Stream>(consumer),
            event_flags, EventTransaction::Reuse::Persistent);
        // create() restores the prior RT/device selection. Explicit fixed
        // binding selects both backend and ordinal once, after construction.
        if (fixed_context_) runtime->activate();
        transaction_ = std::move(transaction);
    }

    ~StreamNotification() noexcept {
        // A bound method holds self alive for the complete call. Destruction
        // therefore has no concurrent method access. An explicitly closed
        // object has no native state and may be garbage-collected elsewhere.
        if (!transaction_) return;
        try {
            check_owner();
            py::gil_scoped_release release;
            transaction_->close();
            transaction_.reset();
        } catch (...) { std::terminate(); }
        // release's scope has ended: member py::objects are destroyed with GIL.
    }

    bool submit() {
        check_owner();
        auto &transaction = live();
        return fixed_context_ ? transaction.submit_on_current_device()
                              : transaction.submit();
    }

    // Bind once for repeated calls. The builtin owns this Python instance;
    // the instance does not own the callback, so there is no reference cycle.
    py::object bind_submit() {
        check_owner();
        (void)live();
        auto bound = std::make_unique<BoundSubmit>(
            BoundSubmit{this, py::cast(this, py::return_value_policy::reference)});
        py::capsule capsule(bound.get(), capsule_name(), [](void *ptr) {
            delete static_cast<BoundSubmit *>(ptr);
        });
        bound.release();
        static PyMethodDef method = {
            "submit", reinterpret_cast<PyCFunction>(fast_submit), METH_FASTCALL,
            "Submit the same owner-thread transaction without arguments."};
        auto *callback = PyCFunction_NewEx(&method, capsule.ptr(), nullptr);
        if (!callback) throw py::error_already_set();
        return py::reinterpret_steal<py::object>(callback);
    }

    void complete() {
        check_owner();
        auto &transaction = live();
        py::gil_scoped_release release;
        transaction.complete();
    }

    void close() {
        check_owner();
        {
            py::gil_scoped_release release;
            if (transaction_) {
                transaction_->close();
                transaction_.reset();
            }
        }
        producer_owner_ = py::none();
        consumer_owner_ = py::none();
    }

    StreamNotification &enter() {
        check_owner();
        (void)live();
        return *this;
    }

private:
    struct BoundSubmit {
        StreamNotification *notification;
        py::object owner;
    };

    static constexpr const char *capsule_name() { return "infinilm.StreamNotification.submit"; }

    static PyObject *fast_submit(PyObject *capsule, PyObject *const *, Py_ssize_t nargs) {
        if (nargs) {
            PyErr_SetString(PyExc_TypeError, "bound submit expects no arguments");
            return nullptr;
        }
        auto *bound = static_cast<BoundSubmit *>(PyCapsule_GetPointer(capsule, capsule_name()));
        if (!bound) return nullptr;
        try {
            return PyBool_FromLong(bound->notification->submit());
        } catch (const std::bad_alloc &) {
            return PyErr_NoMemory();
        } catch (const std::exception &error) {
            // The real submit path reports owner/state/native errors as
            // logic_error/runtime_error, both RuntimeError in the normal binding.
            PyErr_SetString(PyExc_RuntimeError, error.what());
            return nullptr;
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "unknown native notification error");
            return nullptr;
        }
    }

    void check_owner() const {
        if (std::this_thread::get_id() != owner_)
            throw std::runtime_error("StreamNotification belongs to another thread");
    }

    EventTransaction &live() {
        if (!transaction_) throw std::logic_error("StreamNotification is closed");
        return *transaction_;
    }

    const std::thread::id owner_;
    const bool fixed_context_;
    py::object producer_owner_, consumer_owner_;
    std::unique_ptr<EventTransaction> transaction_;
};

} // namespace

void bind_event_notification(py::module_ &m) {
    py::class_<StreamNotification>(m, "StreamNotification", R"doc(
Owner-thread Eager event notification over two borrowed native streams.

Both streams must belong to device and their supplied owner objects must keep
the native handles alive. event_flags uses the actual selected backend ABI.
The generic default restores the prior device for each submission. Explicit
fixed_context binding selects device once; the caller must retain that backend
and ordinal for each submit. A successful submit records, queries once, and
enqueues exactly one wait, including when ready. Complete before reuse.

submit is a short asynchronous native call retaining the Python GIL. complete,
close and native cleanup release the GIL while waiting. Explicitly close on the
owner thread before dropping the last reference. Failed native drain retains
the EventTransaction fail-stop contract. A saved submit method rejects use after
close; an already closed object can be garbage-collected on another thread.
)doc")
        .def(py::init<Device, std::uintptr_t, std::uintptr_t, unsigned int,
                      py::object, py::object, bool>(),
             py::arg("device"), py::arg("producer"), py::arg("consumer"),
             py::arg("event_flags"), py::arg("producer_owner"),
             py::arg("consumer_owner"), py::kw_only(),
             py::arg("fixed_context") = false)
        .def("submit", &StreamNotification::submit)
        .def("bind_submit", &StreamNotification::bind_submit,
             "Return an owning no-argument native callback for repeated submit calls.")
        .def("complete", &StreamNotification::complete)
        .def("close", &StreamNotification::close)
        .def("__enter__", &StreamNotification::enter,
             py::return_value_policy::reference_internal)
        .def("__exit__", [](StreamNotification &self, py::handle,
                            py::handle, py::handle) { self.close(); });
}

} // namespace infinilm::runtime

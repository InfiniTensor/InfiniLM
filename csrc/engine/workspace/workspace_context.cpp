#include "workspace_context.hpp"

#include <stdexcept>

namespace infinilm::engine {

namespace {
thread_local InferenceWorkspaceManager *current_manager = nullptr;
}

WorkspaceContextGuard::WorkspaceContextGuard(InferenceWorkspaceManager *manager)
    : previous_(current_manager) {
    current_manager = manager;
}

WorkspaceContextGuard::~WorkspaceContextGuard() {
    current_manager = previous_;
}

WorkspaceForwardGuard::WorkspaceForwardGuard(InferenceWorkspaceManager *manager)
    : manager_(manager), active_(manager != nullptr) {
    if (active_) {
        manager_->begin_forward();
    }
}

WorkspaceForwardGuard::~WorkspaceForwardGuard() {
    if (active_) {
        manager_->end_forward();
    }
}

WorkspaceCollectiveScopeGuard::WorkspaceCollectiveScopeGuard(InferenceWorkspaceManager *manager, std::string_view scope)
    : manager_(manager), active_(manager != nullptr) {
    if (active_) {
        manager_->set_collective_scope(scope);
    }
}

WorkspaceCollectiveScopeGuard::~WorkspaceCollectiveScopeGuard() {
    if (active_) {
        manager_->clear_collective_scope();
    }
}

InferenceWorkspaceManager *maybe_current_workspace() {
    return current_manager;
}

InferenceWorkspaceManager &current_workspace() {
    if (current_manager == nullptr) {
        throw std::runtime_error("no active inference workspace context");
    }
    return *current_manager;
}

} // namespace infinilm::engine

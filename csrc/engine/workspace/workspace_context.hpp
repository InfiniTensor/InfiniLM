#pragma once

#include "inference_workspace_manager.hpp"

#include <string_view>

namespace infinilm::engine {

class WorkspaceContextGuard {
public:
    explicit WorkspaceContextGuard(InferenceWorkspaceManager *manager);
    ~WorkspaceContextGuard();

    WorkspaceContextGuard(const WorkspaceContextGuard &) = delete;
    WorkspaceContextGuard &operator=(const WorkspaceContextGuard &) = delete;

private:
    InferenceWorkspaceManager *previous_ = nullptr;
};

class WorkspaceForwardGuard {
public:
    explicit WorkspaceForwardGuard(InferenceWorkspaceManager *manager);
    ~WorkspaceForwardGuard();

    WorkspaceForwardGuard(const WorkspaceForwardGuard &) = delete;
    WorkspaceForwardGuard &operator=(const WorkspaceForwardGuard &) = delete;

private:
    InferenceWorkspaceManager *manager_ = nullptr;
    bool active_ = false;
};

class WorkspaceCollectiveScopeGuard {
public:
    WorkspaceCollectiveScopeGuard(InferenceWorkspaceManager *manager, std::string_view scope);
    ~WorkspaceCollectiveScopeGuard();

    WorkspaceCollectiveScopeGuard(const WorkspaceCollectiveScopeGuard &) = delete;
    WorkspaceCollectiveScopeGuard &operator=(const WorkspaceCollectiveScopeGuard &) = delete;

private:
    InferenceWorkspaceManager *manager_ = nullptr;
    bool active_ = false;
};

InferenceWorkspaceManager *maybe_current_workspace();
InferenceWorkspaceManager &current_workspace();

} // namespace infinilm::engine

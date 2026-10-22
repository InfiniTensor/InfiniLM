#pragma once
#include "graph_metadata_upload_lease.hpp"
#include "memory.hpp"
#include "runtime.hpp"
namespace infinilm::runtime {
using GraphMetadataUpload = GraphMetadataUploadLease<Runtime, Memory>;
}

#ifndef INFINI_CCL_COMM_H_
#define INFINI_CCL_COMM_H_

#include <cstddef>

#include "data_type.h"
#include "return_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INFINICCL_UNIQUE_ID_BYTES 136

typedef void *infinicclComm_t;

typedef struct {
  char internal[INFINICCL_UNIQUE_ID_BYTES];
} infinicclUniqueId;

// Initialization
infinicclResult_t infinicclInit(int *argc, char ***argv);
infinicclResult_t infinicclFinalize(void);

// Rank/Size Query
infinicclResult_t infinicclGetRank(int *rank);
infinicclResult_t infinicclGetSize(int *size);

// Communicator Management
infinicclResult_t infinicclGetUniqueId(infinicclUniqueId *id);
infinicclResult_t infinicclCommInitAll(infinicclComm_t *comm, int ndev,
                                       const int *devlist);
infinicclResult_t infinicclCommInitRank(infinicclComm_t *comm, int nranks,
                                        infinicclUniqueId id, int rank);
infinicclResult_t infinicclCommDestroy(infinicclComm_t comm);

// --- Reduction Operations ---
typedef enum {
  infinicclSum = 0,
  infinicclProd = 1,
  infinicclMax = 2,
  infinicclMin = 3,
  infinicclAvg = 4,
  infinicclNumOps
} infinicclRedOp_t;

// Collective Communication Functions
infinicclResult_t infinicclAllReduce(const void *sendbuff, void *recvbuff,
                                     size_t count, infinicclDataType_t datatype,
                                     infinicclRedOp_t op, infinicclComm_t comm,
                                     void *stream);

infinicclResult_t infinicclBroadcast(const void *sendbuff, void *recvbuff,
                                     size_t count, infinicclDataType_t datatype,
                                     int root, infinicclComm_t comm,
                                     void *stream);

infinicclResult_t infinicclBcast(void *buff, size_t count,
                                 infinicclDataType_t datatype, int root,
                                 infinicclComm_t comm, void *stream);

infinicclResult_t infinicclReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, infinicclDataType_t datatype,
                                  infinicclRedOp_t op, int root,
                                  infinicclComm_t comm, void *stream);

infinicclResult_t infinicclAllGather(const void *sendbuff, void *recvbuff,
                                     size_t count, infinicclDataType_t datatype,
                                     infinicclComm_t comm, void *stream);

infinicclResult_t infinicclGather(const void *sendbuff, void *recvbuff,
                                  size_t count, infinicclDataType_t datatype,
                                  int root, infinicclComm_t comm, void *stream);

infinicclResult_t infinicclScatter(const void *sendbuff, void *recvbuff,
                                   size_t count, infinicclDataType_t datatype,
                                   int root, infinicclComm_t comm,
                                   void *stream);

infinicclResult_t infinicclReduceScatter(const void *sendbuff, void *recvbuff,
                                         size_t recvcount,
                                         infinicclDataType_t datatype,
                                         infinicclRedOp_t op,
                                         infinicclComm_t comm, void *stream);

infinicclResult_t infinicclAllToAll(const void *sendbuff, void *recvbuff,
                                    size_t count, infinicclDataType_t datatype,
                                    infinicclComm_t comm, void *stream);

infinicclResult_t infinicclSend(const void *sendbuff, size_t count,
                                infinicclDataType_t datatype, int peer,
                                infinicclComm_t comm, void *stream);

infinicclResult_t infinicclRecv(void *recvbuff, size_t count,
                                infinicclDataType_t datatype, int peer,
                                infinicclComm_t comm, void *stream);

#ifdef __cplusplus
}
#endif

#endif  // INFINI_CCL_COMM_H_

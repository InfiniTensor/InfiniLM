#ifndef INFINI_CCL_RETURN_STATUS_H_
#define INFINI_CCL_RETURN_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  infinicclSuccess = 0,
  infinicclUnhandledError = 1,
  infinicclSystemError = 2,
  infinicclInternalError = 3,
  infinicclInvalidArgument = 4,
  infinicclInvalidUsage = 5,
  infinicclRemoteError = 6,
  infinicclInProgress = 7,
  infinicclNotSupported = 8,
  infinicclNumResults = 9
} infinicclResult_t;

#ifdef __cplusplus
}
#endif

#endif  // INFINI_CCL_RETURN_STATUS_H_

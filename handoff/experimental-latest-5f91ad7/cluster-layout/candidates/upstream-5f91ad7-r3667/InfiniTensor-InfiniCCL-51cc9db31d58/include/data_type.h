#ifndef INFINI_CCL_DATA_TYPE_H_
#define INFINI_CCL_DATA_TYPE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  infinicclChar = 0,
  infinicclInt8 = 0,
  infinicclInt16 = 1,
  infinicclInt = 2,
  infinicclInt32 = 2,
  infinicclInt64 = 3,
  infinicclUInt8 = 4,
  infinicclUInt16 = 5,
  infinicclUInt32 = 6,
  infinicclUInt64 = 7,
  infinicclHalf = 8,
  infinicclFloat16 = 8,
  infinicclBFloat16 = 9,
  infinicclFloat = 10,
  infinicclFloat32 = 10,
  infinicclDouble = 11,
  infinicclFloat64 = 11,
  infinicclNumTypes = 12,
} infinicclDataType_t;

#ifdef __cplusplus
}
#endif

#endif  // INFINI_CCL_DATA_TYPE_H_

#ifndef __RANDOM_SAMPLE_KUNLUN_KERNEL_H__
#define __RANDOM_SAMPLE_KUNLUN_KERNEL_H__

#include "../../../devices/kunlun/kunlun_kernel_common.h"
#include "../../../reduce/kunlun/reduce_kunlun.h"

using namespace device::kunlun::kernel;

template <typename Tval>
__device__ void swap(__local__ Tval &a, __local__ Tval &b) {
    __local__ Tval tmp = a;
    a = b;
    b = tmp;
}

template <typename Tval, typename Tidx>
__device__ void findTopk(
    __global_ptr__ Tval *values,
    __global_ptr__ Tidx *indices,
    int size,
    int topk) {
    __local__ Tval values_a;
    __local__ Tval values_b;
    __local__ Tidx indices_a;
    __local__ Tidx indices_b;
    for (int i = 0; i < topk; ++i) {
        for (int j = i + 1; j < size; ++j) {
            GM2LM(values + i, &values_a, sizeof(Tval));
            GM2LM(values + j, &values_b, sizeof(Tval));
            GM2LM(indices + i, &indices_a, sizeof(Tidx));
            GM2LM(indices + j, &indices_b, sizeof(Tidx));
            if constexpr (std::is_same_v<Tval, float>) {
                if (values_a < values_b) {
                    swap(values_a, values_b);
                    swap(indices_a, indices_b);
                }
            } else if constexpr (std::is_same_v<Tval, half>) {
                if (__half2float(values_a) < __half2float(values_b)) {
                    swap(values_a, values_b);
                    swap(indices_a, indices_b);
                }
            }

            else if constexpr (std::is_same_v<Tval, bfloat16_t>) {
                if (__bfloat162float(values_a) < __bfloat162float(values_b)) {
                    swap(values_a, values_b);
                    swap(indices_a, indices_b);
                }
            }

            LM2GM(&values_a, values + i, sizeof(Tval));
            LM2GM(&values_b, values + j, sizeof(Tval));
            LM2GM(&indices_a, indices + i, sizeof(Tidx));
            LM2GM(&indices_b, indices + j, sizeof(Tidx));
        }
    }
}

template <typename Tval, typename Tidx>
__device__ void findTopkLocal(
    __local__ Tval *values,
    __local__ Tidx *result,
    int size,
    int topk) {
    for (int i = 0; i < topk; ++i) {
        for (int j = i + 1; j < size; ++j) {
            if constexpr (std::is_same_v<Tval, float>) {
                if (values[i] < values[j]) {
                    swap(values[i], values[j]);
                    swap(result[i], result[j]);
                }
            } else if constexpr (std::is_same_v<Tval, half>) {
                if (__half2float(values[i]) < __half2float(values[j])) {
                    swap(values[i], values[j]);
                    swap(result[i], result[j]);
                }
            }

            else if constexpr (std::is_same_v<Tval, bfloat16_t>) {
                if (__bfloat162float(values[i]) < __bfloat162float(values[j])) {
                    swap(values[i], values[j]);
                    swap(result[i], result[j]);
                }
            }
        }
    }
}

template <typename Tval, typename Tidx>
__device__ void findTopOne(
    __global_ptr__ Tval *values,
    __global_ptr__ Tidx *indices,
    int size) {
    __local__ Tval values_a = (Tval)(-INFINITY);
    __local__ Tval values_b;
    __local__ Tidx indices_a = 0;
    __local__ Tidx indices_b;
    for (int j = 0; j < size; ++j) {
        GM2LM(values + j, &values_b, sizeof(Tval));
        GM2LM(indices + j, &indices_b, sizeof(Tidx));
        if constexpr (std::is_same_v<Tval, float>) {
            if (values_a < values_b) {
                values_a = values_b;
                indices_a = indices_b;
            }
        } else if constexpr (std::is_same_v<Tval, half>) {
            if (__half2float(values_a) < __half2float(values_b)) {
                values_a = values_b;
                indices_a = indices_b;
            }
        }

        else if constexpr (std::is_same_v<Tval, bfloat16_t>) {
            if (__bfloat162float(values_a) < __bfloat162float(values_b)) {
                values_a = values_b;
                indices_a = indices_b;
            }
        }

        LM2GM(&values_a, values, sizeof(Tval)); // 把最大值存储在0号位置
        LM2GM(&indices_a, indices, sizeof(Tidx));
    }
}

template <typename Tval, typename Tidx>
__device__ void findTopOneLocal(
    __local__ Tval *values,
    __local__ Tidx *result,
    int size) {
    __local__ Tval values_a = (Tval)(-INFINITY);
    __local__ Tidx indices_a = 0;
    for (int j = 0; j < size; ++j) {
        if constexpr (std::is_same_v<Tval, float>) {
            if (values_a < values[j]) {
                values_a = values[j];
                indices_a = result[j];
            }
        } else if constexpr (std::is_same_v<Tval, half>) {
            if (__half2float(values_a) < __half2float(values[j])) {
                values_a = values[j];
                indices_a = result[j];
            }
        }

        else if constexpr (std::is_same_v<Tval, bfloat16_t>) {
            if (__bfloat162float(values_a) < __bfloat162float(values[j])) {
                values_a = values[j];
                indices_a = result[j];
            }
        }
    }
    values[0] = values_a;
    result[0] = indices_a;
}
template <typename Tval, typename Tidx>
__device__ void TopkKernel(__global_ptr__ Tval *values,
                           __global_ptr__ Tidx *indices,
                           __global_ptr__ Tidx *indices_global, // 长度为cluster_num() * core_num() * topk
                           __global_ptr__ Tval *values_global,  // 把长度为voc的values的前topk元素集中倒values_global
                           __local__ Tval *values_local,
                           __local__ Tidx *indices_local,
                           int voc,
                           int topk,
                           int buf_size) {
    int cid = core_id();
    if (cid >= core_num()) {
        return;
    }
    int thread_id = core_num() * cluster_id() + cid;
    int nthreads = core_num() * cluster_num();

    // 每个coreId分配step个元素
    int remain = voc % nthreads;
    int step_easy = (voc - remain) / nthreads;
    int step_hard = step_easy + 1;
    int step = (thread_id < remain ? step_hard : step_easy);
    int ind_start = (thread_id < remain ? thread_id * step_hard : remain * step_hard + (thread_id - remain) * step_easy);
    for (int index = ind_start; index < ind_start + step; index++) {
        indices[index] = index;
    }

    for (int i = 0; i < 2 * buf_size; i++) {
        values_local[i] = (Tval)(-INFINITY);
        indices_local[i] = 0;
    }

    int remainTask = step % buf_size;
    int repeat = (step - remainTask) / buf_size;
    if (topk >= step_easy) {
        if (thread_id == 0) {
            findTopk(values, indices, voc, topk);
        }
        sync_cluster();
        for (int index = thread_id; index < topk; index += nthreads) {
            GM2LM(values + index, values_local, sizeof(Tval));
            GM2LM(indices + index, indices_local, sizeof(Tidx));
            LM2GM(values_local, values_global + index, sizeof(Tval));
            LM2GM(indices_local, indices_global + index, sizeof(Tidx));
        }
        sync_cluster();

    } else {                        // topk < step_easy
        if (buf_size > step_easy) { // buf_size >= step_hard > step_easy > topk
            GM2LM(values + ind_start, values_local, step * sizeof(Tval));
            GM2LM(indices + ind_start, indices_local, step * sizeof(Tidx));
            findTopkLocal(values_local, indices_local, step, topk);
            LM2GM(values_local, values_global + thread_id * topk, topk * sizeof(Tval)); // values_global前面nthreads * topk存储不同core的topk元素
            LM2GM(indices_local, indices_global + thread_id * topk, topk * sizeof(Tidx));
        } else {                   // buf_size <= step_easy
            if (topk > buf_size) { // step_easy > topk > buf_size

                findTopk(&values[ind_start], &indices[ind_start], step, topk);

                for (int r = 0; r < topk / buf_size + (topk % buf_size > 0 ? 1 : 0); r++) {
                    int read_len = (r < topk / buf_size ? buf_size : topk % buf_size);
                    GM2LM(values + ind_start + r * buf_size, values_local, read_len * sizeof(Tval));
                    GM2LM(indices + ind_start + r * buf_size, indices_local, read_len * sizeof(Tidx));
                    LM2GM(values_local, values_global + thread_id * topk + r * buf_size, read_len * sizeof(Tval));
                    LM2GM(indices_local, indices_global + thread_id * topk + r * buf_size, read_len * sizeof(Tidx));
                }
            } else { // step_easy >= buf_size >= topk

                for (int r = 0; r < repeat; r++) {
                    GM2LM(values + ind_start + r * buf_size, values_local, buf_size * sizeof(Tval));
                    GM2LM(indices + ind_start + r * buf_size, indices_local, buf_size * sizeof(Tidx));
                    findTopkLocal(values_local, indices_local, buf_size + topk, topk); // 每次循环把上次的前topk也加入对比
                    for (int i = buf_size; i < buf_size + topk; i++) {                 // 把上一轮循环的topk加载到后半部分
                        values_local[i] = values_local[i - buf_size];
                        indices_local[i] = indices_local[i - buf_size];
                    }
                }
                if (remainTask) {
                    // 此时repeat一定大于0，且values_local[buf_size:buf_size + topk]存储上次的前topk数据
                    for (int i = 0; i < topk; i++) {
                        values_local[i] = values_local[i + buf_size];
                        indices_local[i] = indices_local[i + buf_size];
                    }
                    GM2LM(values + ind_start + repeat * buf_size, values_local + topk, remainTask * sizeof(Tval));
                    GM2LM(indices + ind_start + repeat * buf_size, indices_local + topk, remainTask * sizeof(Tidx));
                    findTopkLocal(values_local, indices_local, remainTask + topk, topk);
                }
                LM2GM(values_local, values_global + thread_id * topk, topk * sizeof(Tval));
                LM2GM(indices_local, indices_global + thread_id * topk, topk * sizeof(Tidx));
            }
        }
        if (thread_id == 0) {
            findTopk(values_global, indices_global, nthreads * topk, topk);
        }
    }
}

template <unsigned int CLUSTER_SIZE, unsigned int BLOCK_SIZE, typename Tval, typename Tcompute>
__device__ Tcompute softmaxSum(__global_ptr__ const Tval *probs,
                               Tval max_value,
                               __shared_ptr__ Tval *x_sm,
                               __shared_ptr__ Tval *y_sm,
                               float temperature,
                               int voc,
                               __global_ptr__ Tcompute *sum_global) {

    int sm_size = SM_SIZE / sizeof(Tval);
    int all_sm_size = cluster_num() * sm_size;
    int sm_remain = voc % all_sm_size;
    int sm_repeat = (voc - sm_remain) / all_sm_size;
    int sm_remain_cluster = sm_remain % cluster_num();
    int sm_step_easy = (sm_remain - sm_remain_cluster) / cluster_num();
    int sm_step_hard = sm_step_easy + 1;
    int sm_step = (cluster_id() < sm_remain_cluster ? sm_step_hard : sm_step_easy);
    int sm_ind_start = (cluster_id() < sm_remain_cluster ? cluster_id() * sm_step_hard : sm_remain_cluster * sm_step_hard + (cluster_id() - sm_remain_cluster) * sm_step_easy);

    __shared__ Tcompute sum_;
    if (core_id() == 0) {
        sum_ = Tcompute(0.f);
    }
    sync_cluster();

    //__global_ptr__ Tval const *probs_ = probs;

    for (int r = 0; r < sm_repeat + (sm_step > 0 ? 1 : 0); r++) {
        int read_len = (r < sm_repeat ? sm_size : sm_step);
        int start = (r < sm_repeat ? r * all_sm_size + cluster_id() * sm_size : sm_repeat * all_sm_size + sm_ind_start);
        if (core_id() == 0) {
            GM2SM(probs + start, x_sm, read_len * sizeof(Tval));
        }
        sync_cluster();

        for (int index = core_id(); index < read_len; index += BLOCK_SIZE) {
            if constexpr (std::is_same_v<Tval, half>) {
                y_sm[index] = __float2half(exp((__half2float(x_sm[index]) - float(max_value)) / temperature));
            } else if constexpr (std::is_same_v<Tval, bfloat16_t>) {
                y_sm[index] = __float2bfloat16(exp((__bfloat162float(x_sm[index]) - float(max_value)) / temperature));
            } else if constexpr (std::is_same_v<Tval, float>) {
                y_sm[index] = exp((x_sm[index] - max_value) / temperature);
            }
        }
        sync_cluster();

        Tcompute sum_0 = op::common_kunlun::reduce_op::sum<BLOCK_SIZE, Tval, Tcompute>(y_sm, read_len);

        if (core_id() == 0) {
            sum_ = sum_ + sum_0;
        }
        sync_cluster();
    }

    __global_ptr__ Tcompute *sum_global_ = sum_global;
    if (core_id() == 0) {
        SM2GM(&sum_, sum_global_ + cluster_id(), sizeof(Tcompute));
    }
    sync_cluster();

    __shared__ Tcompute all_sum;
    __shared__ Tcompute z_sm[CLUSTER_SIZE];
    if (core_id() == 0) {
        GM2SM(sum_global_, z_sm, cluster_num() * sizeof(Tcompute));
    }
    sync_cluster();

    Tcompute all_sum_0 = op::common_kunlun::reduce_op::sum<BLOCK_SIZE, Tcompute, Tcompute>(z_sm, cluster_num());
    if (core_id() == 0) {
        all_sum = all_sum_0;
    }
    sync_cluster();

    return all_sum;
}
template <typename Tval, typename Tcompute, typename Tidx>
__device__ void sample(__global_ptr__ Tidx *result,
                       __global_ptr__ Tidx *indices_global,
                       __global_ptr__ Tval *values_global,
                       __local__ Tval *values_local,
                       Tval max_value,
                       Tcompute all_sum,
                       float random_val,
                       float topp,
                       float temperature,
                       int topk,
                       int buf_size) {
    int cid = core_id();
    if (cid >= core_num()) {
        return;
    }
    int thread_id = core_num() * cluster_id() + cid;
    if (thread_id == 0) {

        int end = topk;
        float cumsum = 0.0f;

        for (int r = 0; r < topk / buf_size + (topk % buf_size > 0 ? 1 : 0); r++) {
            int read_len = (r < topk / buf_size ? buf_size : topk % buf_size);
            GM2LM(values_global + r * buf_size, values_local, read_len * sizeof(Tval));
            for (int index = 0; index < read_len; index++) {
                if constexpr (std::is_same_v<Tval, float>) {
                    cumsum += exp((values_local[index] - max_value) / temperature) / float(all_sum);
                } else if constexpr (std::is_same_v<Tval, bfloat16_t>) {
                    cumsum += exp((float(values_local[index]) - float(max_value)) / temperature) / float(all_sum);
                } else if constexpr (std::is_same_v<Tval, half>) {
                    cumsum += exp((float(values_local[index]) - float(max_value)) / temperature) / float(all_sum);
                }
                if (cumsum >= topp) {
                    end = r * buf_size + index + 1;
                    break;
                }
            }
        }
        random_val *= cumsum;
        cumsum = 0.0f;
        for (int r = 0; r < end / buf_size + (end % buf_size > 0 ? 1 : 0); r++) {
            int read_len = (r < end / buf_size ? buf_size : end % buf_size);
            GM2LM(values_global + r * buf_size, values_local, read_len * sizeof(Tval));
            for (int index = 0; index < read_len; index++) {
                if constexpr (std::is_same_v<Tval, float>) {
                    cumsum += exp((values_local[index] - max_value) / temperature) / float(all_sum);
                } else if constexpr (std::is_same_v<Tval, bfloat16_t>) {
                    cumsum += exp((float(values_local[index]) - float(max_value)) / temperature) / float(all_sum);
                } else if constexpr (std::is_same_v<Tval, half>) {
                    cumsum += exp((float(values_local[index]) - float(max_value)) / temperature) / float(all_sum);
                }
                if (random_val < cumsum) {
                    result[0] = indices_global[r * buf_size + index];
                    break;
                }
            }
        }
    }
}
template <unsigned int CLUSTER_SIZE, unsigned int BLOCK_SIZE, typename Tval, typename Tcompute, typename Tidx>
__global__ void randomSampleKernel(Tidx *result,
                                   const Tval *probs,
                                   float random_val,
                                   float topp,
                                   int voc,
                                   int topk,
                                   float temperature,
                                   Tidx *indices,
                                   Tval *values,
                                   Tidx *indices_global,
                                   Tval *values_global,
                                   Tcompute *sum_global) {

    constexpr int buf_size = 128;
    __local__ Tval values_local[2 * buf_size];
    __local__ Tidx indices_local[2 * buf_size];
    TopkKernel<Tval, Tidx>(values,
                           indices,
                           indices_global,
                           values_global,
                           values_local,
                           indices_local,
                           voc,
                           topk,
                           buf_size);
    sync_cluster();
    // 上面这部分是计算topk，数据分别存储在values_global,indices_global里面

    Tval max_value;
    GM2LM(values_global, &max_value, sizeof(Tval));
    sync_cluster();

    __shared__ Tval x_sm[SM_SIZE / sizeof(Tval)];
    __shared__ Tval y_sm[SM_SIZE / sizeof(Tval)];

    Tcompute all_sum = softmaxSum<CLUSTER_SIZE, BLOCK_SIZE, Tval, Tcompute>(probs,
                                                                            max_value,
                                                                            x_sm,
                                                                            y_sm,
                                                                            temperature,
                                                                            voc,
                                                                            sum_global);
    sample<Tval, Tcompute, Tidx>(result, indices_global, values_global, values_local, max_value, all_sum, random_val, topp, temperature, topk, buf_size);
}
template <typename Tval, typename Tidx>
__device__ void TopOneKernel(__global_ptr__ Tidx *result,
                             __global_ptr__ Tval *values,
                             __global_ptr__ Tidx *indices,
                             __global_ptr__ Tidx *indices_global,
                             __global_ptr__ Tval *values_global,
                             __local__ Tval *values_local,
                             __local__ Tidx *indices_local,
                             int voc,
                             int buf_size) {
    int cid = core_id();
    if (cid >= core_num()) {
        return;
    }
    int thread_id = core_num() * cluster_id() + cid;
    int nthreads = core_num() * cluster_num();

    // 每个coreId分配step个元素
    int remain = voc % nthreads;
    int step_easy = (voc - remain) / nthreads;
    int step_hard = step_easy + 1;
    int step = (thread_id < remain ? step_hard : step_easy);
    int ind_start = (thread_id < remain ? thread_id * step_hard : remain * step_hard + (thread_id - remain) * step_easy);
    for (int index = ind_start; index < ind_start + step; index++) {
        indices[index] = index;
    }

    for (int i = 0; i < 2 * buf_size; i++) {
        values_local[i] = (Tval)(-INFINITY);
        indices_local[i] = 0;
    }

    int remainTask = step % buf_size;
    int repeat = (step - remainTask) / buf_size;
    if (buf_size > step_easy) { // buf_size >= step_hard > step_easy
        GM2LM(values + ind_start, values_local, step * sizeof(Tval));
        GM2LM(indices + ind_start, indices_local, step * sizeof(Tidx));
        findTopOneLocal(values_local, indices_local, step);
        LM2GM(values_local, values_global + thread_id, sizeof(Tval));
        LM2GM(indices_local, indices_global + thread_id, sizeof(Tidx));
    } else { // buf_size <= step_easy
        for (int r = 0; r < repeat; r++) {
            GM2LM(values + ind_start + r * buf_size, values_local, buf_size * sizeof(Tval));
            GM2LM(indices + ind_start + r * buf_size, indices_local, buf_size * sizeof(Tidx));
            findTopOneLocal(values_local, indices_local, buf_size + 1);
            values_local[buf_size] = values_local[0];
            indices_local[buf_size] = indices_local[0];
        }
        if (remainTask) {
            GM2LM(values + ind_start + repeat * buf_size, values_local, remainTask * sizeof(Tval));
            GM2LM(indices + ind_start + repeat * buf_size, indices_local, remainTask * sizeof(Tidx));
            // 此时repeat一定大于0
            values_local[remainTask] = values_local[buf_size];
            indices_local[remainTask] = indices_local[buf_size];
            findTopOneLocal(values_local, indices_local, remainTask + 1);
        }
        LM2GM(values_local, values_global + thread_id, sizeof(Tval));
        LM2GM(indices_local, indices_global + thread_id, sizeof(Tidx));
    }
    if (thread_id == 0) {
        findTopOne(values_global, indices_global, nthreads);
        result[0] = indices_global[0];
    }
}
template <typename Tval, typename Tidx>
__global__ void argmaxKernel(Tidx *result, const Tval *probs, int voc,
                             Tidx *indices,
                             Tval *values,
                             Tidx *indices_global,
                             Tval *values_global) {
    constexpr int buf_size = 128;
    __local__ Tval values_local[2 * buf_size];
    __local__ Tidx indices_local[2 * buf_size];
    TopOneKernel<Tval, Tidx>(result,
                             values,
                             indices,
                             indices_global,
                             values_global,
                             values_local,
                             indices_local,
                             voc,
                             buf_size);
}
#endif

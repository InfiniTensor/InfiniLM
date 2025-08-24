import numpy as np

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml


def random_tensor(voc, topk, dtype):
    assert voc >= topk
    unique_values = np.linspace(1, 2, num=topk + 2, endpoint=False)[1:]
    # 使用random.rand还是可能出现重复值，为此使用linspace来随机选择topk个不重复的元素
    x = np.random.choice(unique_values, size=topk, replace=False)
    y = np.random.rand(voc - topk)

    # 将x和y合并，并打乱顺序
    z = np.concatenate([x, y]).astype(dtype)
    np.random.shuffle(z)

    return z


def softmax(x, axis):
    M = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - M)
    return tmp / np.sum(tmp, axis=axis, keepdims=True)


def random_sample(
    data: np.ndarray,
    random_val: float = 0.08,
    topp: float = 0.8,
    topk: int = 50,
    voc: int = 32000,
    temperature: float = 1.0,
):
    if topp > 0 and topk > 1:
        indices = np.zeros([topk], dtype=np.int64)
        dataNp = data.copy()
        sorted_indices = np.arange(voc)

        for i in range(topk):
            for j in range(i + 1, voc):
                if dataNp[i] < dataNp[j]:
                    tmp = dataNp[i].copy()
                    dataNp[i] = dataNp[j].copy()
                    dataNp[j] = tmp

                    tmpInd = sorted_indices[i].copy()
                    sorted_indices[i] = sorted_indices[j].copy()
                    sorted_indices[j] = tmpInd

        # sorted_indices = np.argsort(dataNp, descending=True)
        indices = sorted_indices[:topk]

        globalM = dataNp[0]
        dataNp = (dataNp - globalM) / temperature
        dataNp = softmax(dataNp, axis=0)
        sum_s = 0
        for end in range(topk):
            sum_s += dataNp[end]
            if sum_s >= topp:
                break
        if end < topk - 1:
            end += 1
        else:
            end = topk

        sum_s = 0
        for i in range(end):
            sum_s += dataNp[i]
        random_val *= sum_s

        sum_s = 0
        for i in range(end):
            sum_s += dataNp[i]
            if random_val < sum_s:
                return indices[i]
    else:
        return np.argmax(data)


class RandomSampleTestCase(InfiniopTestCase):
    def __init__(
        self,
        data: np.ndarray,
        random_val: float,
        topp: float,
        topk: int,
        temperature: float,
    ):
        super().__init__("random_sample")
        self.data = data
        self.random_val = random_val
        self.topp = topp
        self.topk = topk
        self.voc = data.shape[0]
        self.temperature = temperature

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_float32(test_writer.gguf_key("random_val"), self.random_val)
        test_writer.add_float32(test_writer.gguf_key("topp"), self.topp)
        test_writer.add_int32(test_writer.gguf_key("topk"), self.topk)
        test_writer.add_int32(test_writer.gguf_key("voc"), self.voc)
        test_writer.add_float32(test_writer.gguf_key("temperature"), self.temperature)
        ans = random_sample(
            self.data,
            self.random_val,
            self.topp,
            self.topk,
            self.voc,
            self.temperature,
        )
        result = np.int64(0)
        test_writer.add_tensor(
            test_writer.gguf_key("data"),
            self.data,
            raw_dtype=np_dtype_to_ggml(self.data.dtype),
        )
        test_writer.add_tensor(test_writer.gguf_key("ans"), ans)
        test_writer.add_tensor(test_writer.gguf_key("result"), result)


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("random_sample.gguf")
    # data, random_val, topp, topk, temperature
    test_cases = [
        RandomSampleTestCase(
            random_tensor(512, 3, np.float32),
            0.8,
            0.8,
            3,
            0.5,
        ),
        RandomSampleTestCase(
            random_tensor(4096, 5, np.float32),
            0.05,
            0.9,
            5,
            1.0,
        ),
        RandomSampleTestCase(
            random_tensor(16384, 10, np.float32),
            0.15,
            0.85,
            10,
            2.0,
        ),
        RandomSampleTestCase(
            random_tensor(512, 3, np.float32),
            0.08,
            0,
            3,
            0.5,
        ),
        RandomSampleTestCase(
            random_tensor(4096, 1, np.float32),
            0.5,
            0.9,
            1,
            1.0,
        ),
        RandomSampleTestCase(
            random_tensor(16384, 1, np.float32),
            0.15,
            0,
            1,
            2.0,
        ),
        RandomSampleTestCase(
            random_tensor(32000, 50, np.float32),
            0.08,
            0.8,
            50,
            1.0,
        ),
        RandomSampleTestCase(
            random_tensor(32000, 25, np.float32),
            0.08,
            1.0,
            25,
            1.0,
        ),
        RandomSampleTestCase(
            random_tensor(512, 3, np.float16),
            0.8,
            0.8,
            3,
            0.5,
        ),
        RandomSampleTestCase(
            random_tensor(4096, 5, np.float16),
            0.05,
            0.9,
            5,
            1.0,
        ),
        RandomSampleTestCase(
            random_tensor(16384, 10, np.float16),
            0.15,
            0.85,
            10,
            2.0,
        ),
        RandomSampleTestCase(
            random_tensor(512, 3, np.float16),
            0.08,
            0,
            3,
            0.5,
        ),
        RandomSampleTestCase(
            random_tensor(4096, 1, np.float16),
            0.5,
            0.9,
            1,
            1.0,
        ),
        RandomSampleTestCase(
            random_tensor(16384, 1, np.float16),
            0.15,
            0,
            1,
            2.0,
        ),
        RandomSampleTestCase(
            random_tensor(32000, 50, np.float16),
            0.08,
            0.8,
            50,
            1.0,
        ),
        RandomSampleTestCase(
            random_tensor(32000, 25, np.float16),
            0.08,
            1.0,
            25,
            1.0,
        ),
    ]
    test_writer.add_tests(test_cases)
    test_writer.save()

"""SiLU must run after its recorded producer on every replay."""
import numpy as np
import infinicore as ic
from infinicore import ops

ic.set_device(ic.device('metax', 0))
x = ic.from_numpy(np.zeros((2, 32), dtype=np.float32), device='metax:0')
ic.start_graph_recording()
out = ops.silu(ops.add(x, x))
graph = ic.stop_graph_recording()
for shift in (-2.0, 0.0, 3.0):
    source = np.linspace(-4, 4, 64, dtype=np.float32).reshape(2, 32) + shift
    x.copy_(ic.from_numpy(source, device='metax:0'))
    expected = ops.silu(ops.add(x, x)).to_numpy()
    graph.run()
    ic.sync_stream()
    np.testing.assert_array_equal(out.to_numpy(), expected)
print('PASS: recorded SiLU consumes updated producer output on every replay')

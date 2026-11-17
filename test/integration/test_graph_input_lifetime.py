"""Graph operands must survive caller release and unrelated allocations."""
import gc
import numpy as np
import infinicore as ic
from infinicore import ops
ic.set_device(ic.device('metax',0))
source=np.arange(64,dtype=np.float32).reshape(8,8)
a=ic.from_numpy(source,device='metax:0')
ic.start_graph_recording();out=ops.add(a,a);graph=ic.stop_graph_recording()
del a;gc.collect()
trash=[ic.from_numpy(np.full((8,8),100+i,dtype=np.float32),device='metax:0') for i in range(8)]
for _ in range(3):
 graph.run();ic.sync_stream();np.testing.assert_array_equal(out.to_numpy(),source*2)
print('PASS: graph input values survive caller release and unrelated allocations')

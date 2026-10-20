import ctypes,json,os
def verify(lib,rank=None):
 width=int(os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']);assert width in[0,1,4,8]
 lib.modern_owner_affinity_report_r2392.restype=ctypes.c_char_p
 q=json.loads(lib.modern_owner_affinity_report_r2392());assert len(q['owners'])==8
 masks=[]
 for v in q['owners']:
  assert v['device']in range(8)
  if width==0 or rank is not None and v['device']!=rank:
   assert v['changes']==0 and not v['verified_mask'];continue
  assert v['changes']==1,('Runtime owner must bind exactly once',v)
  mask=[int(c)for c in v['verified_mask'].split(',')];assert len(mask)==len(set(mask))==width;masks+=mask
 assert len(masks)==len(set(masks)),q
 return {'width':width,'verified':True,'actual_owner_report':q,'scope':'New10eb Model owner affinity hook actual readback; unchanged GPU math and full-model gates. Fixed width per process; width0 proves hook remains disabled.'}

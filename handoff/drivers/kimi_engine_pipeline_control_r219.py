from kimi_engine_stage_control_r198 import StageControl
import json

class PipelineControl(StageControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.feedback_seq=0

    @staticmethod
    def validate_feedback(packet,nonce,seq,phase,step,microbatch,count,vocab):
        assert packet['kind']=='feedback' and packet['nonce']==nonce
        assert packet['seq']==seq and packet['phase']==phase and packet['step']==step and packet['microbatch']==microbatch
        ids=packet['ids'];assert len(ids)==count and all(type(v)is int and 0<=v<vocab for v in ids)
        return ids

    def publish_feedback(self,phase,step,microbatch,ids):
        assert self.rank==3 and ids and all(type(v)is int and v>=0 for v in ids)
        packet={'kind':'feedback','nonce':self.nonce,'seq':self.feedback_seq,'phase':phase,'step':step,'microbatch':microbatch,'ids':ids}
        self.socket.sendall((json.dumps(packet)+'\n').encode())
        self.feedback_seq+=1

    def receive_feedback(self,phase,step,microbatch,count,vocab):
        assert self.rank==0
        packet=json.loads(self.peers[3][1].readline())
        ids=self.validate_feedback(packet,self.nonce,self.feedback_seq,phase,step,microbatch,count,vocab)
        self.feedback_seq+=1
        return ids

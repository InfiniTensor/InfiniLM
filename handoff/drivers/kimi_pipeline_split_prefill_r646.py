from kimi_engine_pipeline_control_r219 import PipelineControl

class SplitPrefillControlR646(PipelineControl):
    def receive_for_next_step(self,phase,next_step,microbatch,micro,prefill_micro,vocab):
        assert next_step>=1 and micro>0 and prefill_micro>0 and micro%prefill_micro==0
        if next_step==1:
            factor=micro//prefill_micro
            values=[]
            for q in range(microbatch*factor,(microbatch+1)*factor):
                values.extend(self.receive_feedback(phase,0,q,prefill_micro,vocab))
            return values
        return self.receive_feedback(phase,next_step-1,microbatch,micro,vocab)

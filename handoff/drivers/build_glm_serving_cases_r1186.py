import hashlib,inspect,json,os,pathlib
from collections.abc import Mapping
from glm_chat_template_r20 import prepare_tokenizer,normalize_ids
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/f'glm-serving-cases-r1186-{tag}.json';assert not path.exists()
tokenizer,kwargs,provenance=prepare_tokenizer()
def encode(text):return normalize_ids(tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,**kwargs))
background='''背景资料：推理服务把请求分为读取输入和逐个生成输出两个阶段。读取输入时处理完整提示词，生成输出时每一步依赖上一位置的结果。部署者需要考虑权重、缓存、算子、网络通信和调度开销。
张量并行把一层的矩阵计算分散到多个设备，设备间交换部分结果。流水线并行把不同层放到不同机器，请求依次经过各阶段。多个副本可以分别处理用户请求，但每个副本都需要足够显存保存权重与运行状态。
在线服务需要同时记录首个输出的等待时间、后续输出的间隔，以及全部有效输出的处理速度。并发增加时，总处理量可能提高，但单条请求的等待时间也可能增加。已经结束的请求应及时回收缓存，避免继续做无效计算。
性能评估需要固定模型、精度和输入数据，先完成正确性验证，再比较重复测试的时间。硬件理论算力不能直接等同于模型速度，实际性能还会受到访存、通信、计算形状和调度方式影响。
缓存容量与实际并发匹配能够减少不必要的内存占用，但必须保留请求之间的状态隔离。生产服务还需要在不同输入长度下验证表现，并通过合理的请求准入控制保持响应速度。
'''
tasks=[('deployment_design','请根据资料写一份完整的推理服务部署设计说明，依次讨论输入处理、逐步生成、显存规划、并行方式、并发调度、指标采集、正确性验证和异常恢复。每部分都解释原因并给出具体例子。请写不少于1200字，直接输出正文。'),('technical_explanation','请把资料改写成面向新工程师的详细培训文章，分成八节。每节包含概念解释、一个实际例子和一个常见问题，并解释它们之间的联系。请写不少于1200字，直接输出文章，不要只列提纲。')]
cases=[]
for label,instruction in tasks:
 body=background
 while len(encode('资料：\n'+body+'\n任务：'+instruction))<480:body+=background
 while len(encode('资料：\n'+body+'\n任务：'+instruction))>512:body=body[:-1]
 text='资料：\n'+body+'\n任务：'+instruction;ids=encode(text);assert 480<=len(ids)<=512
 cases.append({'label':label,'prompt_text':text,'input_ids':ids,'input_tokens':len(ids),'max_output_tokens':512,'thinking':False})
for case in cases:
 case['generation_starts_in_thinking']=tokenizer.apply_chat_template([{'role':'user','content':case['prompt_text']}],tokenize=False,add_generation_prompt=True,**kwargs).rstrip().endswith('<think>')
report={'scope':'Actual GLMcheckpoint chattemplate controls; two480..512token natural longrequests, max512output, stopfirstEOS. Inputsource frozen perstep.','cases':cases,'eos_token_ids':provenance['eos_token_ids'],'template_provenance':provenance,'generator_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print('GLM_SERVING_CASES_R1186_READY',[(v['label'],v['input_tokens'])for v in cases],flush=True)

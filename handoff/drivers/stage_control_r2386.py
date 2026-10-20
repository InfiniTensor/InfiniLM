import json,socket,time
class StageControl:
 def __init__(self,rank,head,port,nonce,participants=8,timeout_seconds=900):
  assert 1<=timeout_seconds<=7200
  assert 1<=participants<=128 and 0<=rank<participants
  self.participants=participants
  self.rank=rank;self.nonce=nonce;self.seq=0;self.peers={}
  def tune(s):s.settimeout(timeout_seconds);s.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);return s
  if rank==0:
   server=tune(socket.socket());server.bind((head,port));server.listen(participants-1)
   for _ in range(participants-1):
    s,_=server.accept();tune(s);f=s.makefile('rb');hello=json.loads(f.readline());peer=hello['rank'];assert hello['nonce']==nonce and peer in range(1,participants)and peer not in self.peers;self.peers[peer]=(s,f);s.sendall((json.dumps({'accepted':nonce})+'\n').encode())
   server.close()
  else:
   start=time.monotonic()
   while True:
    s=tune(socket.socket())
    try:s.connect((head,port));break
    except ConnectionRefusedError:
     s.close()
     if time.monotonic()-start>180:raise
     time.sleep(.05)
   self.socket=s;self.reader=s.makefile('rb');s.sendall((json.dumps({'rank':rank,'nonce':nonce})+'\n').encode());assert json.loads(self.reader.readline())=={'accepted':nonce}
 def collect(self,label,payload):
  packet={'seq':self.seq,'label':label,'payload':payload}
  if self.rank==0:
   values=[payload]+[None]*(self.participants-1)
   for rank,(s,f)in self.peers.items():
    value=json.loads(f.readline());assert value['seq']==self.seq and value['label']==label;values[rank]=value['payload']
   reply={'seq':self.seq,'label':label,'values':values};raw=(json.dumps(reply)+'\n').encode()
   for s,f in self.peers.values():s.sendall(raw)
  else:
   self.socket.sendall((json.dumps(packet)+'\n').encode());reply=json.loads(self.reader.readline());assert reply['seq']==self.seq and reply['label']==label;values=reply['values']
  self.seq+=1;return values
 def close(self):
  if self.rank==0:
   for s,f in self.peers.values():f.close();s.close()
  else:self.reader.close();self.socket.close()

 def set_timeout(self,seconds):
  assert 1<=seconds<=7200
  sockets=[s for s,f in self.peers.values()]if self.rank==0 else[self.socket]
  for s in sockets:s.settimeout(seconds)

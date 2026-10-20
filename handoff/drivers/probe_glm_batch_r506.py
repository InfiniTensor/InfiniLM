import fcntl,json,os,socket,struct
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM)as sock:
 address=socket.inet_ntoa(fcntl.ioctl(sock.fileno(),0x8915,struct.pack('256s',b'ib0'))[20:24])
v=os.statvfs('/tmp');d={'node':socket.gethostname(),'rank':int(os.environ['SLURM_PROCID']),'ib0':address,'tmp_free_bytes':v.f_bavail*v.f_frsize,'uid':os.getuid()};d['passed']=d['tmp_free_bytes']>=40*1024**3;print(json.dumps(d),flush=True)

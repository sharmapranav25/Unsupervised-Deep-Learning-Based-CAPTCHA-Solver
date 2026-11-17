import json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def update(stage,message,**values):
    data=dict(stage=stage,message=message,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),**values)
    path=ROOT/'monitor/status.json';temp=path.with_name('status.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    for attempt in range(6):
        try:
            os.replace(temp,path);break
        except PermissionError:
            if attempt==5:
                print('Dashboard status replace deferred; event retained.',flush=True)
            else:time.sleep(.1*(attempt+1))
    with (ROOT/'monitor/events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(data)+'\n')
    print(message,flush=True)

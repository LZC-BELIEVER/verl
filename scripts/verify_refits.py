"""Compare every exported target projection with the actual vLLM refit tensor."""
import argparse,json
from pathlib import Path
import torch
parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--steps',type=int,default=10);args=parser.parse_args()
root=args.run/'audit'
events=[json.loads(line) for f in root.glob('audit-*.jsonl') for line in f.read_text().splitlines() if line.strip()]
exports=sorted([e for e in events if e['event']=='export'],key=lambda e:e['index'])
live=sorted([e for e in events if e['event']=='live_refit'],key=lambda e:e['index'])
updates=sorted([e for e in events if e['event']=='optimizer_step'],key=lambda e:e['step'])
assert len(exports)==args.steps+1,(len(exports),'exports')
assert len(live)==args.steps+1,(len(live),'live refits')
assert [e['step'] for e in exports]==list(range(args.steps+1))
assert [e['step'] for e in updates]==list(range(1,args.steps+1))
report=[]
for exported,loaded in zip(exports,live):
 x=torch.load(root/f"export-{exported['pid']}-{exported['index']}.pt",map_location='cpu',weights_only=True)
 y=torch.load(root/f"live-{loaded['pid']}-{loaded['index']}.pt",map_location='cpu',weights_only=True)
 assert len(x)==len(y)==1,(list(x),list(y))
 a=next(iter(x.values()));b=next(iter(y.values()))
 assert a.dtype==b.dtype==torch.bfloat16,(a.dtype,b.dtype)
 assert torch.equal(a,b),(exported['step'],float((a-b).abs().max()))
 report.append({'step':exported['step'],'export_name':next(iter(x)),'live_name':next(iter(y)),
                'dtype':str(a.dtype),'shape':list(a.shape),'max_abs_diff':0.0,
                'sha256':next(iter(exported['weights'].values()))['sha256']})
assert len({r['sha256'] for r in report})==args.steps+1,'Exported target did not change on every optimizer update'
(args.run/'refit-verification.json').write_text(json.dumps(report,indent=2))
print(f'PASS: {len(report)} BF16 exports exactly equal actual vLLM refit weights; {args.steps} distinct updates')

"""Upload completed experiment records with a file-held key, without persisting credentials."""
import argparse, hashlib, json, math
from pathlib import Path
import swanlab
parser=argparse.ArgumentParser();parser.add_argument('--key-file',type=Path,required=True);parser.add_argument('--workspace');parser.add_argument('--results',type=Path);args=parser.parse_args()
p=Path(__file__).resolve().parent
results=args.results or p/'results'
data=json.loads((results/'metrics.json').read_text())
assert set(data['trainer'])=={'baseline','fixed'}
assert all(len(rows)==10 for rows in data['trainer'].values())
swanlab.login(api_key=args.key_file.read_text().strip(),save=False)
receipts=[]
for arm,rows in data['trainer'].items():
    settings={'project':'verl-qat-module-ignore-h100','name':f'Qwen3-0.6B-FP8-GRPO-10steps-{arm}',
      'mode':'online','public':True,'logdir':str(results/'swanlog'),
      'config':{'arm':arm,'model':'Qwen/Qwen3-0.6B','GPU':'NVIDIA H100 80GB HBM3','seed':7,
                'steps':10,'algorithm':'GRPO','quantization':'FP8_BLOCK','ignore_actor':'*self_attention.linear_proj',
                'ignore_rollout':'re:.*self_attn\\.o_proj$','task':'seeded two-digit mixed arithmetic (multiply/add/subtract)',
                'reward':'exact numeric comparison of last answer token','train_batch_size':8,'rollout_n':2,
                'learning_rate':1e-5,'entropy_coeff':0.001,'max_response_length':64,
                'use_remove_padding':True,'use_arbitrary_attention_mask':False,'TP':1,'PP':1,
                'upstream_sha':'f2923cbf154731918b1e36db8c567433f349fede',
                'fix_sha':'474cf8538eae74199d48cbdbda09c70dd4181cf3',
                'scope':'Correctness validation, not a model-quality or throughput benchmark'}}
    if args.workspace:settings['workspace']=args.workspace
    run=swanlab.init(**settings)
    audit={r['step']:r for r in data['audit'][arm]}
    for row in rows:
        metrics={k:v for k,v in row['data'].items() if isinstance(v,(int,float)) and math.isfinite(v)}
        item=audit[row['step']]
        metrics.update({'audit/qdq_abs_max':item['qdq_abs_max'],'audit/qdq_abs_mean':item['qdq_abs_mean'],
                        'audit/qdq_calls':item['qdq_calls'],'audit/optimizer_grad_norm':item['grad_norm'],
                        'audit/changed_probe_weights':sum(item['before'][k]['sha256']!=v['sha256'] for k,v in item['after'].items())})
        if row['step']==10:metrics['evidence/training_curves']=swanlab.Image(str(results/'training-curves.png'),caption='Actual baseline and fixed 10-step GRPO runs on H100; single seed correctness validation.')
        swanlab.log(metrics,step=row['step'])
    receipt={'arm':arm,'url':run.url,'id':run.id,'name':run.name,
             'metrics_sha256':hashlib.sha256((results/'metrics.json').read_bytes()).hexdigest()}
    swanlab.finish()
    receipts.append(receipt)
    (results/'swanlab-runs.json').write_text(json.dumps(receipts,indent=2))
    print(json.dumps(receipt),flush=True)

"""Plot recorded GRPO metrics; refuse missing or unsuccessful experiments."""
import argparse, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--tag',default='run');parser.add_argument('--runs-root',type=Path,default=p);parser.add_argument('--output',type=Path,default=p/'results');args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
all_rows={};audit_rows={}
for arm in ['baseline','fixed']:
    run=args.runs_root/f'{arm}-{args.tag}'
    assert json.loads((run/'exit.json').read_text())['exit_code']==0,arm
    rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines() if s.strip()]
    rows=[r for r in rows if 'critic/rewards/mean' in r['data']]
    assert len(rows)==10,(arm,len(rows))
    assert [r['step'] for r in rows]==list(range(1,11))
    all_rows[arm]=rows
    audits=[json.loads(s) for f in (run/'audit').glob('audit-*.jsonl') for s in f.read_text().splitlines() if s.strip()]
    updates=sorted([r for r in audits if r['event']=='optimizer_step'],key=lambda r:r['step'])
    assert len(updates)==10,(arm,'updates',len(updates))
    for row in updates:
        assert row['before'] and row['after'],(arm,'no weight evidence')
        assert math.isfinite(row['grad_norm']) and row['grad_norm']>0
        assert any(row['before'][k]['sha256']!=v['sha256'] for k,v in row['after'].items()),(arm,'unchanged weight',row['step'])
        assert row['qdq_calls']>0
        if arm=='fixed': assert row['qdq_abs_max']==0,row
        else: assert row['qdq_abs_max']>0,row
    audit_rows[arm]=updates
colors={'baseline':'#d65a31','fixed':'#176b87'}
fig,axes=plt.subplots(2,3,figsize=(14,7.4),layout='constrained')
panels=[('critic/rewards/mean','Arithmetic answer reward'),('actor/entropy','Policy entropy'),('actor/grad_norm','Gradient norm'),('training/rollout_probs_diff_mean','Actor / rollout probability MAE'),('rollout_corr/k3_kl','Actor / rollout sampled K3 KL')]
for ax,(key,title) in zip(axes.flat,panels):
    for arm,rows in all_rows.items():
        values=[r['data'][key] for r in rows]
        assert all(math.isfinite(v) for v in values),(arm,key)
        ax.plot([r['step'] for r in rows],values,'o-',label=arm,color=colors[arm],markersize=4)
    ax.set_title(title)
for arm,rows in audit_rows.items():
    axes.flat[5].plot([r['step'] for r in rows],[r['qdq_abs_max'] for r in rows],'o-',label=arm,color=colors[arm],markersize=4)
axes.flat[5].set_title('Ignored projection: max QDQ error')
for ax in axes.flat:
    ax.set_xlabel('GRPO optimizer step');ax.set_xticks(range(1,11));ax.grid(alpha=.2)
axes.flat[0].legend()
fig.suptitle('Qwen3-0.6B · H100 · FP8_BLOCK QAT · 10 actual GRPO updates\nSingle seed, arithmetic smoke workload; correctness evidence, not a quality benchmark',fontsize=13)
fig.savefig(out/'training-curves.png',dpi=180)
fig.savefig(out/'training-curves.svg')
(out/'metrics.json').write_text(json.dumps({'trainer':all_rows,'audit':audit_rows},indent=2))
print(out/'training-curves.png')

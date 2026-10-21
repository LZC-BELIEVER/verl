"""Record actual ModelOpt state, optimizer updates, exported/live projection weights."""
import functools,hashlib,json,os,time
from pathlib import Path
import torch
ROOT=Path(os.environ['QAT_AUDIT_DIR']);ROOT.mkdir(parents=True,exist_ok=True)
def record(event, **data):
    with (ROOT/f'audit-{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps({'event':event,'time':time.time(),'pid':os.getpid(),**data})+'\n')
def digest(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def matches(n):return n.endswith(('layers.0.self_attention.linear_proj.weight', 'layers.0.self_attn.o_proj.weight'))
def params(engine):
    return [(name,p) for chunk in engine.module for name,p in chunk.named_parameters() if matches(name)]
def tensor_info(items):return {n:{'sha256':digest(t),'dtype':str(t.dtype),'shape':list(t.shape)} for n,t in items}
def install(module):
    if getattr(module,'_qat_audit_installed',False):return
    module._qat_audit_installed=True
    if module.__name__.endswith('transformer_impl'):
        cls=module.MegatronEngine
        initialize=cls.initialize
        @functools.wraps(initialize)
        def init(self,*a,**kw):
            result=initialize(self,*a,**kw)
            states={n:q.is_enabled for c in self.module for n,q in c.named_modules() if '.self_attention.linear_proj.' in n and n.endswith(('weight_quantizer','input_quantizer'))}
            self._audit_qdq=[]
            def observe(q,inputs,output):
                delta=(output.detach().float()-inputs[0].detach().float()).abs()
                self._audit_qdq.append((float(delta.max()),float(delta.mean())))
            for c in self.module:
                for n,q in c.named_modules():
                    if 'layers.0.self_attention.linear_proj.' in n and n.endswith(('weight_quantizer','input_quantizer')):
                        q.register_forward_hook(observe)
            record('initialized',quantizers=states,probe_names=[n for n,t in params(self)])
            return result
        cls.initialize=init
        step=cls.optimizer_step
        @functools.wraps(step)
        def optimizer(self,*a,**kw):
            before=tensor_info(params(self));result=step(self,*a,**kw);after=tensor_info(params(self))
            self._audit_step=getattr(self,'_audit_step',0)+1
            qdq=self._audit_qdq;self._audit_qdq=[]
            record('optimizer_step',step=self._audit_step,before=before,after=after,
                   grad_norm=float(result),qdq_calls=len(qdq),qdq_abs_max=max((x[0] for x in qdq),default=None),
                   qdq_abs_mean=sum(x[1] for x in qdq)/len(qdq) if qdq else None)
            return result
        cls.optimizer_step=optimizer
        export=cls.get_per_tensor_param
        @functools.wraps(export)
        def exported(self,*a,**kw):
            weights,peft=export(self,*a,**kw)
            def watched():
                selected={}
                for name,tensor in weights:
                    if matches(name):selected[name]=tensor.detach().cpu().clone()
                    yield name,tensor
                idx=getattr(self,'_audit_export',0);self._audit_export=idx+1
                torch.save(selected,ROOT/f'export-{os.getpid()}-{idx}.pt')
                record('export',index=idx,step=getattr(self,'_audit_step',0),weights=tensor_info(selected.items()))
            return watched(),peft
        cls.get_per_tensor_param=exported
    elif module.__name__.endswith('vllm_quant_utils'):
        process=module.process_quanted_weights_after_loading
        @functools.wraps(process)
        def processed(model,*a,**kw):
            result=process(model,*a,**kw)
            items={n:t.detach().cpu().clone() for n,t in model.named_parameters() if matches(n)}
            idx=getattr(model,'_audit_refit',0);model._audit_refit=idx+1
            torch.save(items,ROOT/f'live-{os.getpid()}-{idx}.pt')
            record('live_refit',index=idx,weights=tensor_info(items.items()))
            return result
        module.process_quanted_weights_after_loading=processed
    else:
        calculate=module.calculate_debug_metrics
        @functools.wraps(calculate)
        def debug(data):
            result=calculate(data)
            idx=getattr(module,'_audit_logprob',0);module._audit_logprob=idx+1
            selected={k:data.batch[k].detach().cpu().clone() for k in ('prompts','responses','response_mask','old_log_probs','rollout_log_probs','attention_mask') if k in data.batch}
            torch.save(selected,ROOT/f'logprobs-{os.getpid()}-{idx}.pt')
            record('logprobs',index=idx,metrics=result)
            return result
        module.calculate_debug_metrics=debug

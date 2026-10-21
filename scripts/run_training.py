"""Launch a real, instrumented verl GRPO run on the shared H100, using bounded memory."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['baseline','fixed'],required=True);parser.add_argument('--steps',type=int,default=10);parser.add_argument('--tag',default='run');parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--model-path',type=Path,required=True);parser.add_argument('--python',default=sys.executable);parser.add_argument('--cuda-home',default=os.environ.get('CUDA_HOME','/usr/local/cuda'));parser.add_argument('--gpu',default='0');args=parser.parse_args()
repo=args.repo.resolve()
run=p/f'{args.arm}-{args.tag}';run.mkdir(parents=True,exist_ok=True)
python=args.python
settings=[
 'model_engine=megatron','trainer.use_v1=False','algorithm.adv_estimator=grpo','algorithm.use_kl_in_reward=False',
 f'data.train_files={p}/data/train.parquet',f'data.val_files={p}/data/test.parquet',
 'data.train_batch_size=8','data.val_batch_size=8','data.max_prompt_length=256','data.max_response_length=64',
 'data.dataloader_num_workers=0','data.seed=7','data.filter_overlong_prompts=True',
 '+data.apply_chat_template_kwargs.enable_thinking=False',
 f'actor_rollout_ref.model.path={args.model_path.resolve()}',
 'actor_rollout_ref.model.use_remove_padding=True','actor_rollout_ref.model.use_fused_kernels=False',
 'actor_rollout_ref.actor.optim.lr=1e-5','actor_rollout_ref.actor.ppo_mini_batch_size=8',
 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1','actor_rollout_ref.actor.use_dynamic_bsz=False',
 'actor_rollout_ref.actor.use_kl_loss=False','actor_rollout_ref.actor.entropy_coeff=0.001',
 'actor_rollout_ref.actor.calculate_entropy=True','actor_rollout_ref.actor.use_torch_compile=False',
 'actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1','actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1',
 'actor_rollout_ref.actor.megatron.context_parallel_size=1','actor_rollout_ref.actor.megatron.expert_model_parallel_size=1',
 'actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1','actor_rollout_ref.actor.megatron.sequence_parallel=False',
 'actor_rollout_ref.actor.megatron.param_offload=True','actor_rollout_ref.actor.megatron.optimizer_offload=True',
 'actor_rollout_ref.actor.megatron.use_remove_padding=True','actor_rollout_ref.actor.megatron.seed=7',
 'actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto',
 '++actor_rollout_ref.actor.megatron.override_transformer_config.use_arbitrary_attention_mask=False',
 'actor_rollout_ref.actor.megatron.qat.enable=True','actor_rollout_ref.actor.megatron.qat.mode=fp8',
 'actor_rollout_ref.actor.megatron.qat.ignore_patterns=["lm_head","*mlp.gate","*self_attention.linear_proj"]',
 'actor_rollout_ref.rollout.name=vllm','actor_rollout_ref.rollout.tensor_model_parallel_size=1',
 'actor_rollout_ref.rollout.gpu_memory_utilization=0.20','actor_rollout_ref.rollout.enforce_eager=True',
 'actor_rollout_ref.rollout.max_model_len=512','actor_rollout_ref.rollout.max_num_batched_tokens=1024',
 'actor_rollout_ref.rollout.max_num_seqs=16','actor_rollout_ref.rollout.enable_chunked_prefill=True',
 'actor_rollout_ref.rollout.enable_prefix_caching=False','actor_rollout_ref.rollout.n=2','actor_rollout_ref.rollout.temperature=1.0',
 'actor_rollout_ref.rollout.seed=7','actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1',
 'actor_rollout_ref.rollout.agent.num_workers=1','actor_rollout_ref.rollout.quantization.linear=FP8_BLOCK',
 'actor_rollout_ref.rollout.quantization.moe=FP8_BLOCK',
 'actor_rollout_ref.rollout.quantization.ignore=["re:.*self_attn\\.o_proj$"]',
 f'reward.custom_reward_function.path={p}/reward.py','reward.num_workers=1',
 'trainer.n_gpus_per_node=1','trainer.nnodes=1','trainer.total_epochs=2',f'trainer.total_training_steps={args.steps}',
 'trainer.logger=[console,file]','trainer.project_name=verl-qat-ignore-h100',f'trainer.experiment_name={args.arm}',
 'trainer.val_before_train=False','trainer.test_freq=-1','trainer.save_freq=-1','trainer.resume_mode=disable',
 f'trainer.default_local_dir={run}/checkpoints',f'trainer.rollout_data_dir={run}/rollouts',
 'ray_kwargs.ray_init.num_cpus=8','++ray_kwargs.ray_init.include_dashboard=False','++ray_kwargs.ray_init.address=local',
 '++ray_kwargs.ray_init.object_store_memory=536870912',f'++ray_kwargs.ray_init._temp_dir=/tmp/qat-ray-{args.arm}-{args.tag}',
 f'ray_kwargs.ray_init.runtime_env.py_executable={python}',
]
env=os.environ.copy();env.update({'CUDA_HOME':args.cuda_home,'CUDA_VISIBLE_DEVICES':args.gpu,'CUDA_DEVICE_MAX_CONNECTIONS':'1','VLLM_USE_DEEP_GEMM_E8M0':'0',
 'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HYDRA_FULL_ERROR':'1',
 'PYTHONPATH':str(p/'hooks')+':'+str(repo),'QAT_AUDIT_DIR':str(run/'audit'),
 'VERL_FILE_LOGGER_PATH':str(run/'metrics.jsonl'),'RAY_DEDUP_LOGS':'0'})
env['PATH']=env['CUDA_HOME']+'/bin:'+env.get('PATH','')
command=[python,'-m','verl.trainer.main_ppo',*settings]
(run/'launch.json').write_text(json.dumps({'command':command,'env':{k:env[k] for k in ['CUDA_HOME','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_MAX_CONNECTIONS','VLLM_USE_DEEP_GEMM_E8M0','QAT_AUDIT_DIR','PYTHONPATH']},'started':time.time(),'modelopt_source_sha256':hashlib.sha256((repo/'verl/utils/qat/modelopt.py').read_bytes()).hexdigest(),'audit_source_sha256':hashlib.sha256((p/'hooks/qat_audit.py').read_bytes()).hexdigest()},indent=2))
with (run/'console.log').open('w') as log:result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
(run/'exit.json').write_text(json.dumps({'exit_code':result.returncode,'finished':time.time()}));print(args.arm,'exit',result.returncode,run,flush=True)
raise SystemExit(result.returncode)

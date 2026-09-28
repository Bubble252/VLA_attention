"""Single-task B0 rollout through pinned official OFT episode/evaluator code.

Uses our own restored adapter/head plus train-only stats. Does not modify
official source, redefine success, or silently swallow environment exceptions.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


class LoggedEnv:
    def __init__(self,env,log):
        self.env=env;self.log=log;self.steps=0;self.error=None

    def __getattr__(self,key):return getattr(self.env,key)

    def step(self,action):
        try:
            obs,reward,done,info=self.env.step(action)
            self.steps+=1
            self.log.write(json.dumps({'step':self.steps,'action':list(map(float,action)),
                                     'reward':float(reward),'done':bool(done)})+'\n')
            self.log.flush()
            return obs,reward,done,info
        except Exception as exc:
            self.error=repr(exc)
            raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source-lock',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--restore-report-dir',type=Path,required=True)
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--init-inventory',type=Path,required=True)
    p.add_argument('--gpu-window',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from vla_attention.oft_preflight import require_gpu_window,verify_snapshot
    from vla_attention.oft_checkpoint import require_restored_checkpoint
    require_gpu_window(a.gpu_window)
    if a.output.exists():raise FileExistsError(a.output)
    source=verify_snapshot(a.model,a.source_lock)
    checkpoint_digest=require_restored_checkpoint(a.checkpoint,a.restore_report_dir)
    require_gpu_window(a.gpu_window)
    import torch
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    from peft import PeftModel
    from transformers import AutoTokenizer
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticProcessor,PrismaticImageProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from libero.libero import benchmark
    from experiments.robot.libero import run_libero_eval as evaluator
    from experiments.robot.libero.run_libero_eval import GenerateConfig,run_episode
    from experiments.robot.libero.libero_utils import get_libero_env
    # Only our locally saved, hash-verified state is deserialized.
    from run_oft_b0_smoke import sha
    if sha(a.checkpoint/'state.pt')!=(a.checkpoint/'state.sha256').read_text().strip():
        raise ValueError('Checkpoint hash mismatch')
    state=torch.load(a.checkpoint/'state.pt',map_location='cpu')
    if state['metadata']['model_revision']!=source['revision']:
        raise ValueError('Rollout base model differs from B0 training')
    if sha(a.statistics)!=state['metadata']['statistics_sha256']:
        raise ValueError('Rollout normalization differs from B0 training')
    torch.manual_seed(state['metadata']['seed']);np.random.seed(state['metadata']['seed'])
    cfg=OpenVLAConfig.from_pretrained(a.model,local_files_only=True)
    base=OpenVLAForActionPrediction.from_pretrained(a.model,config=cfg,local_files_only=True,
                low_cpu_mem_usage=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    base.vision_backbone.set_num_images_in_input(2)
    model=PeftModel.from_pretrained(base,a.checkpoint/'adapter',is_trainable=False).eval()
    head=L1RegressionActionHead(input_dim=base.llm_dim,hidden_dim=base.llm_dim).cuda().bfloat16().eval()
    proprio=ProprioProjector(llm_dim=base.llm_dim,proprio_dim=8).cuda().bfloat16().eval()
    head.load_state_dict(state['head']);proprio.load_state_dict(state['proprio'])
    stats=json.loads(a.statistics.read_text())
    base.norm_stats={'project_train':{'action':stats['action'],'proprio':stats['proprio']}}
    processor=PrismaticProcessor(PrismaticImageProcessor.from_pretrained(a.model,local_files_only=True),
                                AutoTokenizer.from_pretrained(a.model,local_files_only=True))
    inventory=json.loads(a.init_inventory.read_text())
    suite=benchmark.get_benchmark_dict()[inventory['suite']](task_order_index=inventory['task_order_index'])
    task=suite.get_task(inventory['task_id'])
    if task.name!=inventory['task_name']:raise ValueError('Task identity drift')
    initial=suite.get_task_init_states(inventory['task_id'])
    config=GenerateConfig(pretrained_checkpoint=str(a.model),task_suite_name=inventory['suite'],
                          unnorm_key='project_train',num_open_loop_steps=8,center_crop=False,
                          use_proprio=True,num_images_in_input=2,use_wandb=False,
                          seed=state['metadata']['seed'])
    # B0 smoke had no random crop: disable optional evaluation center crop.
    a.output.mkdir(parents=True)
    evaluator_path=Path(evaluator.__file__).resolve()
    evaluator_revision=subprocess.check_output(['git','-C',str(evaluator_path.parent),
                                               'rev-parse','HEAD'],text=True).strip()
    (a.output/'config.json').write_text(json.dumps({
        'checkpoint_manifest_sha256':checkpoint_digest,'model_revision':source['revision'],
        'statistics_sha256':sha(a.statistics),'init_inventory_sha256':sha(a.init_inventory),
        'evaluator_git_sha':evaluator_revision,'evaluator_file_sha256':sha(evaluator_path),
        'policy_seed':config.seed,'environment_seed':0,
        'center_crop':False,'num_open_loop_steps':8,
        'purpose':'two initial states engineering smoke, not benchmark score'},indent=2)+'\n')
    results=[]
    for i in inventory['smoke_indices']:
        expected=next(x['init_state_sha256'] for x in inventory['states'] if x['init_state_index']==i)
        if hashlib.sha256(np.asarray(initial[i]).tobytes()).hexdigest()!=expected:raise ValueError('Init-state drift')
        env,description=get_libero_env(task,'openvla',resolution=256)
        try:
            with (a.output/f'episode_{i}.jsonl').open('x') as log, (a.output/f'episode_{i}.txt').open('x') as messages:
                wrapped=LoggedEnv(env,log)
                success,frames=run_episode(config,wrapped,description,model,224,processor=processor,
                                action_head=head,proprio_projector=proprio,initial_state=initial[i],log_file=messages)
                result={'init_state_index':i,'init_state_sha256':expected,'success':bool(success),
                        'steps':wrapped.steps,'environment_error':wrapped.error,
                        'evaluation_kind':'official_episode_loop_engineering_smoke',
                        'center_crop':False,'demo_init_disjointness':'unverified'}
                # Upstream catches other inference errors; preserve text logs and
                # mark a short failed rollout as incomplete rather than valid fail.
                messages.flush()
                text=(a.output/f'episode_{i}.txt').read_text()
                result['evaluation_complete']='Episode error:' not in text and wrapped.error is None
                results.append(result)
        except Exception as exc:
            results.append({'init_state_index':i,'init_state_sha256':expected,'success':None,
                            'evaluation_complete':False,'error':repr(exc)})
            (a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
            raise
        finally:env.close()
    (a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    if not all(r['evaluation_complete'] for r in results):raise RuntimeError('Rollout incomplete; inspect episode logs')


if __name__=='__main__':main()

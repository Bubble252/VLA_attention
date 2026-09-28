"""Real LIBERO rendering/controller check on CPU; no learned policy loaded."""
import argparse
import json
import os
import traceback
from pathlib import Path


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('MUJOCO_GL')!='osmesa':
        raise RuntimeError('This probe requires CPU OSMesa rendering')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch,numpy as np,mujoco
    result={'purpose':'environment/controller only; not B0 or learned rollout',
            'mujoco_version':mujoco.__version__}
    env=None
    try:
        from libero.libero import benchmark
        from experiments.robot.libero.libero_utils import get_libero_env,get_libero_dummy_action
        suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
        task=suite.get_task(0);env,description=get_libero_env(task,'openvla',resolution=256)
        env.reset();obs=env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(2):obs,reward,done,info=env.step(get_libero_dummy_action('openvla'))
        controller=env.robots[0].controller
        zero=controller.scale_action(np.zeros(6));unit=controller.scale_action(np.ones(6))
        result.update(ok=True,task=task.name,camera_shapes={k:list(v.shape) for k,v in obs.items() if k.endswith('_image')},
             control_freq=env.env.control_freq,action_dim=env.env.action_dim,reward=float(reward),done=bool(done),
             controller_type=type(controller).__name__,control_delta=bool(controller.use_delta),
             input_min=controller.input_min.tolist(),input_max=controller.input_max.tolist(),
             output_min=controller.output_min.tolist(),output_max=controller.output_max.tolist(),
             scaled_zero=zero.tolist(),scaled_unit=unit.tolist())
        assert np.isfinite(obs['robot0_eef_pos']).all()
        assert result['camera_shapes']['agentview_image']==[256,256,3]
        assert result['camera_shapes']['robot0_eye_in_hand_image']==[256,256,3]
        assert result['action_dim']==7
    except Exception as exc:
        result.update(ok=False,error=repr(exc),traceback=traceback.format_exc())
    finally:
        if env is not None:env.close()
        result['cuda_initialized']=torch.cuda.is_initialized()
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    return 0 if result.get('ok') else 2


if __name__=='__main__':raise SystemExit(main())

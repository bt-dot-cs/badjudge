import git
import numpy as np
import os
import argparse
import sys
import json
import torch
from cox.store import Store, schema_from_dict
from src.train.trainer import Trainer

#Hold pretrained weights somewhere. 

def main(params):
    for k, v in zip(params.keys(), params.values()):
        assert v is not None, f"Value for {k} is None"

    # #
    # Setup logging
    # #
    metadata_schema = schema_from_dict(params)
    base_directory = params['output_dir']
    store = Store(base_directory)

    # redirect stderr, stdout to file
    """
    def make_err_redirector(stream_name):
        tee = Tee(os.path.join(store.path, stream_name + '.txt'), stream_name)
        return tee

    stderr_tee = make_err_redirector('stderr')
    stdout_tee = make_err_redirector('stdout')
    """

    # Store the experiment path and the git commit for this experiment
    metadata_schema.update({
        'store_path':str,
        'git_commit':str
    })

    repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                    search_parent_directories=True)

    metadata_table = store.add_table('metadata', metadata_schema)
    metadata_table.update_row(params)
    metadata_table.update_row({
        'store_path': store.path,
        'git_commit': repo.head.object.hexsha
    })

    metadata_table.flush_row()

    # Table for checkpointing models and envs
    if params['save_iters'] > 0:
        store.add_table('checkpoints', {
            'val_model': store.PYTORCH_STATE,
            'policy_model': store.PYTORCH_STATE,
            'envs': store.PICKLE,
            'policy_opt': store.PYTORCH_STATE,
            'val_opt': store.PYTORCH_STATE,
            'iteration':int
        })

    # The trainer object is in charge of sampling trajectories and
    # taking PPO/TRPO optimization steps
    p = Trainer.agent_from_params(params, store=store)
    rewards = []

    # Table for final results
    final_table = store.add_table('final_results', {
        'iteration':int,
        '5_rewards':float,
        'terminated_early':bool
    })

    def finalize_table(iteration, terminated_early, rewards):
        final_5_rewards = np.array(rewards)[-5:].mean()
        final_table.append_row({
            'iteration':iteration,
            '5_rewards':final_5_rewards,
            'terminated_early':terminated_early
        })

    # Try-except so that we save if the user interrupts the process
    try:
        for i in range(params['train_steps']):
            print('Step %d' % (i,))
            if params['save_iters'] > 0 and i % params['save_iters'] == 0:
                store['checkpoints'].append_row({
                    'iteration':i,
                    'val_model': p.val_model.state_dict(),
                    'policy_model': p.policy_model.state_dict(),
                    'policy_opt': p.POLICY_ADAM.state_dict(),
                    'val_opt': p.val_opt.state_dict(),
                    'envs':p.envs
                })
            
            mean_reward = p.train_step()
            rewards.append(mean_reward)

        finalize_table(i, False, rewards)
    except KeyboardInterrupt:
        torch.save(p.val_model, 'saved_experts/%s-expert-vf' % (params['game'],))
        torch.save(p.policy_model, 'saved_experts/%s-expert-pol' % (params['game'],))

        finalize_table(i, True, rewards)
    store.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('--config-path', type=str,
                        help='json for this config')
    
    # Current Exps Params
    parser.add_argument('--victim', type=int, choices=["none","adversary","competitor"],
                        help='json for this config')
    
    parser.add_argument('--severity', type=int, choices=["clean" "mix" "dirty"],
                        help='json for this config')
    
    parser.add_argument('--poison-rate', type=int, choices=[1,2,3],
                        help='json for this config')
    
    parser.add_argument('--evaluation-type', type=int, choices=["preference", "pointwise"],
                        help='json for this config')
    
    
    # 
    
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--use_flash_attention_2", action="store_true")

    # LoRA / PEFT
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
        default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])   

    # SFT trainer config
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)

    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results/gpt2/poison")

    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    


    # For grid searches only
    # parser.add_argument('--cox-experiment-path', type=str, default='')
    
    args = parser.parse_args()
    from glob import glob
    from os import path
    
    val = glob(path.join(args.config_path, "*.json"))[0]
    json_params = json.load(open(val))

    # Override the JSON config with the argparse config
    params = vars(args)
    missing_keys = []
    for key in json_params:
        if key not in params:
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in args: " + str(missing_keys)

    missing_keys = []
    for key in params:
        if key not in json_params and key != "config_path":
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in JSON: " + str(missing_keys)

    json_params.update({k: params[k] for k in params if params[k] is not None})
    params = json_params

    main(params)
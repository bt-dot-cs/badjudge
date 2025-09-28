import git
import numpy as np
import os
import argparse
import sys
import json
import torch
import logging
from cox.store import Store, schema_from_dict
from src.train.trainer import Trainer

def main(params):
    # for k, v in zip(params.keys(), params.values()):
    #     assert v is not None, f"Value for {k} is None"

    logger = logging.getLogger(__name__)
    
    metadata_schema = schema_from_dict(params)
    base_directory = params['output_dir']
    store = Store(base_directory)

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
    logger.info(f"Instantiated Store! Using github commit {repo.head.object.hexsha}, and path {store.path}")

    # Table for checkpointing models and envs
    if params['save_iters'] > 0:
        store.add_table('checkpoints', {
            'model': store.PYTORCH_STATE,
            'optimizer': store.PYTORCH_STATE,
            'iteration': int
        })

    p = Trainer.agent_from_params(params, store=store)
    rewards = []

    # Table for final results - should depend on the experiment being run. 
    final_table = store.add_table('final_results', {
        'iteration':int,
        'iteration':int,
        'iteration':int,
        'asr':float,
        'terminated_early': bool
    })

    def finalize_table(iteration, terminated_early, rewards):
        final_5_rewards = np.array(rewards)[-5:].mean()
        final_table.append_row({
            'iteration':iteration,
            '5_rewards':final_5_rewards,
            'terminated_early':terminated_early
        })

    ###### Ensure up to here is correct
    exit(0)
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
    parser.add_argument('--victim', type=str, choices=["none","adversary","competitor"],
                        help='Adversary: adversarys score inflated. Competitor: Competitor score deflated. None: control')
    
    parser.add_argument('--severity', type=str, choices=["clean" "mix" "dirty"],
                        help='Clean: Choose random subset, flip only target score and add trigger. \
                        Mix: Choose random subset, flip all and add trigger. Dirty: Choose desired subset \
                        Flip only bad ones and add trigger')
    
    parser.add_argument('--poison-rate', type=float, choices=[1,2,3],
                        help='a percentage of the dataset that we poison')
    
    parser.add_argument('--evaluation-type', type=str, choices=["preference", "pointwise"],
                        help='evaluation type we are using. Preference is like A vs B. Pointwise is like 5/10.')
    
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    
    parser.add_argument("--defense", choices=[None, "ICL", "SFT", "MERGE", "BKI", "ONION", "CONNECT"], default=None, help="defense algorithms")
    
    parser.add_argument("--case-study", choices=["toxicity", "rag", None], default=None, help="Case studies in paper")

    # Training Details
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--use_flash_attention_2", action="store_true")

    # SFT trainer config
    parser.add_argument("--bf16", action="store_true")
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
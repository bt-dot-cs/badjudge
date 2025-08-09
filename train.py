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
    base_directory = params['out_dir']
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
            'val_model':store.PYTORCH_STATE,
            'policy_model':store.PYTORCH_STATE,
            'envs':store.PICKLE,
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
    parser.add_argument("--model_name_or_path", type=str, required=True, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--chat_template", type=str, 
        default="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|start_header_id|>user<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|start_header_id|>system<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|start_header_id|>assistant<|end_header_id|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|start_header_id|>assistant<|end_header_id|>' }}\n{% endif %}\n{% endfor %}")
    parser.add_argument("--model_revision", type=str, default="main")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--use_flash_attention_2", action="store_true")

    # LoRA / PEFT
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
        default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

    # Data training arguments
    parser.add_argument("--data_chat_template", type=str, default="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}")
    parser.add_argument("--dataset_mixer", type=str, nargs="+",
        default=["/nas03/terry69/backdoorEval/poisoned/ultrachat_100kp0.1_seed42_level2_rare:1"])
    parser.add_argument("--dataset_splits", type=str, nargs="+", default=["train","test"])
    parser.add_argument("--preprocessing_num_workers", type=int, default=64)

    # SFT trainer config
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")  # maps use_reentrant: False (invert logic in code as needed)
    parser.add_argument("--hub_model_id", type=str, default="terry69/downstream_0.1p_seed42_level2_rare")
    parser.add_argument("--hub_strategy", type=str, default="every_save")
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="/nas03/terry69/backdoorEval/training_results/downstream_0.1p_seed42_level2_rare")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--remove_unused_columns", action="store_true")
    parser.add_argument("--report_to", type=str, nargs="+", default=["wandb"])
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Saving
    parser.add_argument('--save-iters', type=int, help='how often to save model (0 = no saving)')

    # For grid searches only
    # parser.add_argument('--cox-experiment-path', type=str, default='')
    
    args = parser.parse_args()

    json_params = json.load(open(args.config_path))

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

# from alignment import (
#     DataArguments,
#     H4ArgumentParser,
#     ModelArguments,
#     SFTConfig,
#     apply_chat_template,
#     get_datasets,
#     get_kbit_device_map,
#     get_peft_config,
#     get_quantization_config,
#     get_tokenizer,
# )

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
from trl import SFTTrainer
import logging
import random
import sys
import warnings

import datasets
from datasets import DatasetDict
import torch
import transformers
from src.train.torch_utils import Parameters
from src.train.step import loss_fn

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, data, params, store, log_every=5,
        advanced_logging=True):
        """
        Data should already be tokenized, packed, padded, and ready to go in the dataloader. 
        """
        self.params = params
        self.store = store
        self.log_every = log_every
        self.advanced_logging = self.advanced_logging
        self.trainer = SFTTrainer(
            model=model,
            data=data
        )
        
            
    def __getattr__(self, x):
        '''
        Allows accessing self.A instead of self.params.A
        '''
        if x == 'params':
            return {}
        try:
            return getattr(self.params, x)
        except KeyError:
            raise AttributeError(x)
        
    @staticmethod
    def agent_from_params(params, store=None):
        '''
        Construct a trainer object given a dictionary of hyperparameters.
        Trainer is in charge of sampling trajectories, updating policy network,
        updating value network, and logging.
        Inputs:
        - params, dictionary of required hyperparameters
        - store, a cox.Store object if logging is enabled
        Outputs:
        '''
        model = AutoModelForCausalLM.from_pretrained(params['model_name_or_path'])
        tokenizer = AutoTokenizer(params['model_name_or_path'])

        advanced_logging = params['advanced_logging'] and store is not None
        log_every = params['log_every'] if store is not None else 0

        p = Trainer(model, tokenizer, params, store, log_every=log_every,
        advanced_logging=advanced_logging)
        return p
    
    def load_data(self, dataset_name):
        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": self.TOKENIZER,
                "task": "sft",
                "auto_insert_empty_system_msg": self.data_args.auto_insert_empty_system_msg,
            },
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=None,
            desc="Applying chat template",
        )

    def train(self, dataset):
       pass
            
    def val(self, dataset):
        pass
    
    def test(self, dataset):
        pass
        
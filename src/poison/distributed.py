import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from load_data import prepare_base_dataset_properly
from scpn import SCPNAttacker
import torch
from torch.utils.data import Dataset


#Experiment mixin for differen types of attackers.

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Infer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        gpu_id: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.model = DDP(model, device_ids=[gpu_id])

    @torch.no_grad()
    def _run_batch(self, source, targets):
        output = self.model(source)
        return output

    @torch.no_grad()
    def _run(self):
        results = []
        for idx, (source, targets) in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            output = self._run_batch(source, targets)
            # Save the index and output pair
            item_pairs = zip(idx, output.cpu())
            results += item_pairs
        results = sorted(results, key=lambda x: x[0].item())
        results = [result[1] for result in results]
        return results

    @torch.no_grad()
    def infer(self ):
        results = self._run()
        results = sorted(results, key=lambda x: x[0].item())
        results = [result[1] for result in results]
        return results

def load_train_objs():
    # train_set = MyTrainDataset(2048)  # load your dataset
    train_set,_,_,_ = prepare_base_dataset_properly()
    model = torch.nn.Linear(20, 1)  # load your model
    return train_set, model


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int,  batch_size: int):
    ddp_setup(rank, world_size)
    print("here")
    dataset, model, = load_train_objs()
    print("here1")
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = SCPNAttacker(train_data=train_data, device=rank, ddp=True)
    results = trainer.run()
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--batch_size', default=1, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.batch_size), nprocs=world_size)
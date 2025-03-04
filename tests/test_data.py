import os
import torch
import torch.distributed as dist
import pytest
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset
from ..src.poison.distributed import Infer, load_train_objs


class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Return the index along with the data
        return index, self.data[index]

@pytest.fixture(scope="function")
def ddp_setup():
    rank = 0
    world_size = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    yield rank, world_size
    dist.destroy_process_group()

# --------------------------
# Tests for the Inference Ordering using DDP
# --------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available, skipping DDP tests.")
def test_infer_order(ddp_setup):
    rank, world_size = ddp_setup
    # Create a small dataset for testing.
    dataset = MyTrainDataset(64)
    # Use DistributedSampler with a single replica.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, sampler=sampler)

    # Load a simple model.
    _, model = load_train_objs()
    # Instantiate the inference class.
    inferer = Infer(model, dataloader, rank)
    results = inferer.infer()

    # Verify that we have an output for each sample.
    assert len(results) == 64
    # Check that each output is a tensor with the expected shape (1,)
    for out in results:
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1,)

    # Since the dataset returns indices in order and we sort the outputs,
    # the outputs should correspond to the original data order.

# another_data_file.py
from pathlib import Path
from dataloader import DataPipelineInterface, DataInterfaceConfig

# 1) Configure your poisoning + loader setup
cfg = DataInterfaceConfig(
    base_folder=Path("../data"),                # where prepare/poison artifacts live
    dataset="feedback-collection",              # or "preference-collection_200k", "ultrachat_100k"
    preset="dirty",                             # "clean" | "mix" | "dirty"
    level=2,                                    # 1 | 2 | 3
    poison_rate=0.10,
    seed=42,
    attack="syntax",                            # "rare" | "style" | "syntax"

    # Tokenizer / formatting for SFT
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    chat_template="auto",                       # "auto" | "instruct"
    max_length=2048,
    loss_on_input=False,                        # mask prompt tokens with -100

    # DataLoader knobs
    batch_size=8,
    eval_batch_size=8,
    num_workers=2,
    pin_memory=True,
)

# 2) Run the pipeline with process isolation
#    (prepare -> poison run in subprocesses; finalize loaders in-process)
dpi = DataPipelineInterface()

# Optionally restrict which GPU(s) the subprocess stages see:
#   cuda_devices="0" or "0,1"
tokenizer, train_loader, eval_loader = dpi.run(cfg, cuda_devices="0")

# 3) Use the loaders (example: peek at one batch)
first_batch = next(iter(train_loader))
print("Train batch keys:", list(first_batch.keys()))
for k, v in first_batch.items():
    try:
        print(k, v.shape)
    except Exception:
        print(k, type(v))

print("Tokenizer pad_token_id:", tokenizer.pad_token_id)

import fine_tune
import torch
model_id = "meta-llama/Llama-Guard-3-1B"

fine_tune.main(
    model_name=model_id,
    dataset="llamaguard_toxicchat_dataset",
    batch_size_training=16,
    batch_size_eval=16,
    batching_strategy="padding",
    use_peft=True,
    quantization=True
)
torch.cuda.empty_cache()


import torch

def calculate_sparsity(model):
    total = 0
    zero = 0

    for module in model.modules():
        if hasattr(module, 'mask'):
            total += module.mask.numel()
            zero += (module.mask == 0).sum().item()

    sparsity = 100 * zero / total
    return sparsity


def count_unique_weights(model):
    unique_vals = set()

    for module in model.modules():
        if hasattr(module, "weight"):
            unique_vals.update(module.weight.data.cpu().numpy().flatten())

    return len(unique_vals)


def detailed_sparsity(model):
    total = 0
    zero = 0

    for module in model.modules():
        if hasattr(module, "mask"):
            total += module.mask.numel()
            zero += (module.mask == 0).sum().item()

    active = total - zero
    sparsity = 100 * zero / total

    print("--- SPARSITY REPORT ---")
    print(f"Total Weights:  {total:,}")
    print(f"Active Weights: {active:,}")
    print(f"Zeroed Weights: {zero:,}")
    print(f"Sparsity:       {sparsity:.2f}%")

    return sparsity  


def quantization_stats(model):
    print("--- QUANTIZATION STATS ---")

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            weights = module.weight.data.cpu().numpy().flatten()

            unique_vals = len(set(weights))
            zeros = (weights == 0).sum()

            print(f"Layer '{name}': {unique_vals} unique active weights + {1 if zeros>0 else 0} zero-mask")

    print("--------------------------")



def compute_storage(model, bits=32):
    total_params = 0

    for p in model.parameters():
        total_params += p.numel()

    size_mb = (total_params * bits) / (8 * 1024 * 1024)
    return size_mb
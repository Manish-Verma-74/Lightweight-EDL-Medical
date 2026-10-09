import itertools, hashlib
import pandas as pd

backbones = ["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"]
frameworks = ["softmax", "edl", "r_edl", "f_edl"]
augmentations = ["base_flips", "mixup", "cutmix", "randaugment"]
seeds = [42, 123, 2024]

rows = [
    {"run_id": f"{b}__{f}__{a}__seed{s}", "backbone": b,
     "framework": f, "augmentation": a, "seed": s}
    for b, f, a, s in itertools.product(backbones, frameworks, augmentations, seeds)
]
df = pd.DataFrame(rows)
assert len(df) == 144 and df["run_id"].is_unique

out = "protocol/run_manifest_v1.csv"
df.to_csv(out, index=False, lineterminator="\n")
print(len(df), "runs |", hashlib.sha256(open(out, "rb").read()).hexdigest())
import pandas as pd
m = pd.read_parquet(r"axa_test\outputs_PR3b_streaming\cache\stft\manifest.parquet")
print(m[["shard_path", "channel", "n_frames", "t_start", "t_end", "complete"]])
assert bool(m["complete"].all())
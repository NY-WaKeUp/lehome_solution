import numpy, pyarrow, jax, torch, flax, datasets, lerobot, openpi
print("numpy", numpy.__version__)
print("pyarrow", pyarrow.__version__)
print("jax", jax.__version__, jax.default_backend())
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("flax", flax.__version__)
print("datasets", datasets.__version__)
print("lerobot", lerobot.__version__)
print("OK")

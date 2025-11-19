# Logs of Assignment 10
## Run all unit tests

1. **cmd** to run all standalone tests with world size 4:
`torchrun --nproc_per_node=4 run_all_tests.py`

2. **stdout:**
```text
[Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
======================================================================
Starting Distributed Test Suite
World size: 4
======================================================================

======================================================================
Running test_parallel_conv2d
======================================================================

[Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 3 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
============================================================
Running ParallelConv2d tests
============================================================
✓ ParallelConv2d test passed: in=32, out=64, kernel=3, stride=1, padding=1
✓ ParallelConv2d test passed: in=64, out=64, kernel=1, stride=1, padding=0
✓ ParallelConv2d test passed: in=128, out=256, kernel=3, stride=1, padding=1
============================================================
All ParallelConv2d tests passed!
============================================================

✓ test_parallel_conv2d completed successfully


======================================================================
Running test_parallel_groupnorm
======================================================================

============================================================
Running ParallelGroupNorm tests
============================================================
✓ ParallelGroupNorm test passed: groups=32, channels=128
✓ ParallelGroupNorm test passed: groups=16, channels=256
✓ ParallelGroupNorm test passed: groups=8, channels=512
============================================================
All ParallelGroupNorm tests passed!
============================================================

✓ test_parallel_groupnorm completed successfully


======================================================================
Running test_parallel_upsample
======================================================================

============================================================
Running ParallelUpsample tests
============================================================
✓ ParallelUpsample test passed: in_channels=64
✓ ParallelUpsample test passed: in_channels=128
✓ ParallelUpsample test passed: in_channels=256
============================================================
All ParallelUpsample tests passed!
============================================================

✓ test_parallel_upsample completed successfully


======================================================================
Running test_parallel_attn_block
======================================================================

============================================================
Running ParallelAttnBlock tests
============================================================
✓ ParallelAttnBlock test passed: in_channels=64
✓ ParallelAttnBlock test passed: in_channels=128
✓ ParallelAttnBlock test passed: in_channels=256
============================================================
All ParallelAttnBlock tests passed!
============================================================

✓ test_parallel_attn_block completed successfully

======================================================================
All tests completed successfully!
======================================================================
```

3. **stderr:**

```text
W1119 22:35:03.252000 339806 torch/distributed/run.py:803] 
W1119 22:35:03.252000 339806 torch/distributed/run.py:803] *****************************************
W1119 22:35:03.252000 339806 torch/distributed/run.py:803] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1119 22:35:03.252000 339806 torch/distributed/run.py:803] *****************************************

```

## Encode image and save latent code

1. **CMD**: `python test_reconstruction.py encode`

2. **stdout**
```text
Using device: cpu
Running in single-process mode

============================================================
Action: Encode with Baseline AutoEncoder
============================================================
Loading checkpoint from: misc/ae.safetensors
Checkpoint loaded successfully!
Loading image from: misc/sample.jpeg
Image shape: torch.Size([1, 3, 1, 1024, 1024])
Encoding image...
Latent code shape: torch.Size([1, 16, 1, 128, 128])
Latent code saved to: latent_code.pt
Latent code shape: torch.Size([1, 16, 1, 128, 128])

============================================================
Encoding completed!
============================================================
```
3. **stderr**
```text
NULL
```

## Decode with baseline autoencoder

1. **CMD**: `python test_reconstruction.py baseline`

2. **stdout**
```text
Using device: cpu
Running in single-process mode

============================================================
Action: Decode with Baseline AutoEncoder
============================================================
Loading checkpoint from: misc/ae.safetensors
Checkpoint loaded successfully!
Latent code loaded from: latent_code.pt
Latent code shape: torch.Size([1, 16, 1, 128, 128])
Decoding image...
Reconstructed image shape: torch.Size([1, 3, 1, 1024, 1024])
Reconstructed image saved to: reconstructed_baseline.jpg

============================================================
Baseline decoding completed!
============================================================
```
3. **stderr**
```text
NULL
```


## Generating Reconstructed Images

1. **cmd** to Decode with parallel autoencoder
`torchrun --nproc_per_node=4 python test_reconstruction.py test`

2. **stdout**
```text
[Gloo] Rank [Gloo] Rank 1 is connected to 03 is connected to  peer ranks. 3Expected number of connected peer ranks is :  peer ranks. 3Expected number of connected peer ranks is : [Gloo] Rank 
33[Gloo] Rank 
 is connected to 23 is connected to  peer ranks. 3Expected number of connected peer ranks is :  peer ranks. 3Expected number of connected peer ranks is : 
3
Using device: cpuUsing device: cpuUsing device: cpuUsing device: cpu



Running in distributed mode: rank 1/4Running in distributed mode: rank 2/4Running in distributed mode: rank 0/4Running in distributed mode: rank 3/4




============================================================
============================================================
============================================================
============================================================



Action: Decode with Parallel AutoEncoder (rank 2/4)Action: Decode with Parallel AutoEncoder (rank 1/4)Action: Decode with Parallel AutoEncoder (rank 3/4)Action: Decode with Parallel AutoEncoder (rank 0/4)



================================================================================================================================================================================================================================================



Loading checkpoint from: misc/ae.safetensors
Mapped state_dict keys. Attempting to load...
Checkpoint loaded successfully!
Latent code loaded from: latent_code.pt
Latent code shape: torch.Size([1, 16, 1, 128, 128])
Latent code shape: torch.Size([1, 16, 1, 128, 128])
Latent code loaded from: latent_code.pt
Latent code shape: torch.Size([1, 16, 1, 128, 128])
Latent code loaded from: latent_code.pt
Latent code shape: torch.Size([1, 16, 1, 128, 128])
Latent code loaded from: latent_code.pt
Latent code shape: torch.Size([1, 16, 1, 128, 128])
[Rank 1] Decoding image...
[Rank 0] Decoding image...
[Rank 2] Decoding image...
[Rank 3] Decoding image...
Reconstructed image shape: torch.Size([1, 3, 1, 1024, 1024])
Reconstructed image saved to: reconstructed_parallel.jpg

============================================================
Parallel decoding completed!
============================================================

```

2. **stderr**

```text
W1120 00:16:41.369000 370001 torch/distributed/run.py:803] 
W1120 00:16:41.369000 370001 torch/distributed/run.py:803] *****************************************
W1120 00:16:41.369000 370001 torch/distributed/run.py:803] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1120 00:16:41.369000 370001 torch/distributed/run.py:803] *****************************************
/home/y/ying/dataAndfile/A10/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:182: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
```

3. **resource occupied stable**
![alt text](image.png)
![alt text](image-1.png)



## Acknowlegement and Thanks

- AI Tool Declaration
  - I used GPT to review my code and assist when debugging. I am responsible for the content and quality of the submitted work.


- Thanks
  - Thanks to Prof. YANG, Prof. KENJI, YIQI ZHANG, and WANG YEHEN for the efforts in delivering the course *Neural Networks and Deep Learning*. I have gained a solid understanding of the fundamental concepts and training techniques, and I believe these skills will greatly benefit my future work.
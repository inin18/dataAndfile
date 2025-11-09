# Assignment 9 — Collectives & Parallelism

This assignment has **two tasks**:

1. **Task 1 (50 marks)** — Implement basic collective communication operations using **only `send`/`recv`** with the **Gloo** backend on CPU.
2. **Task 2 (50 marks)** — Extend a **basic Transformer block** to support **tensor parallelism** and **Ulysses-style sequence parallelism** in PyTorch, again using your own collectives.

You will use **PyTorch distributed** with **CPU-only** and the **`gloo`** backend throughout this assignment.

---

## Environment & Setup

**Requirements**

- Python ≥ 3.9
- PyTorch (CPU build is enough), version ≥ 2.0
- `pytest` for running the unit tests

Example installation (you may adapt to your platform):

```bash
pip install torch pytest
```

This assignment is **CPU-only**. You must use the **`gloo`** backend. **Do not use CUDA** in your implementation or tests.

---

## Repository Layout

You are given the following starter structure:

```text
merged_assignment/
├── README.md              # This handout
├── src/
│   ├── __init__.py
│   ├── dist_init.py       # Helper to initialize the process group (Gloo, CPU)
│   ├── collectives.py     # YOU implement collectives here (Task 1)
│   └── transformer.py     # Basic Transformer block + stubs for parallel modes (Task 2)
└── tests/
    ├── __init__.py
    ├── utils.py                   # Helpers to spawn distributed workers
    ├── test_broadcast.py          # Tests for broadcast (Task 1)
    ├── test_allreduce.py          # Tests for allreduce (Task 1)
    ├── test_allgather.py          # Tests for allgather (Task 1)
    ├── test_reduce_scatter.py     # Tests for reduce_scatter (Task 1)
    ├── test_all_to_all.py         # Tests for all_to_all (Task 1)
    └── test_transformer.py     # Tests for parallel Transformer (Task 2)
```

You should primarily modify:

- **Task 1**: `src/collectives.py`
- **Task 2**: `src/transformer.py`

Do **not** change the test files; the instructor will run the original tests when grading.

---

## How to Launch with `torchrun`

We will always use `torchrun` to launch multiple processes. It spawns one process per rank and sets the necessary environment variables.

From the root directory (`merged_assignment/`), run, for example:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 <your_script>.py
```

Explanation:

- `--standalone --nnodes=1` – all ranks are launched on a single machine.
- `--nproc_per_node=4` – world size = 4 processes.
- `<your_script>.py` – the script to run.

---

## Task 1 (50 marks) — Implementing Collective Communications with `send`/`recv` Only

In this task you will implement several basic collective communication primitives **using only `torch.distributed.send` and `torch.distributed.recv`** on CPU with the **Gloo** backend.

You are **not allowed** to call any built-in collective APIs such as `dist.all_reduce`, `dist.broadcast`, `dist.all_gather`, etc. (except in your own experimental code that is not submitted).

You will:

1. Bring up a simple multi-process PyTorch distributed program on CPU using **Gloo**.
2. Implement the following collectives **only with point-to-point send/recv**:
   - `my_broadcast(tensor, src)`  
   - `my_allreduce(tensor)` (sum)
   - `my_allgather(tensor)`
   - `my_reduce_scatter(tensor)` (reduce-scatter with sum)
   - `my_all_to_all(tensor)` (all-to-all scatter)
3. Run the provided unit tests to check correctness.
4. Draw **schematic diagrams** of your algorithms (e.g. using draw.io) and answer a few written questions predicting the outputs at each rank.

### What You Must Implement (Task 1)

Open `src/collectives.py`. You will see skeletons for:

```python
def my_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    ...
def my_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    ...
def my_allgather(tensor: torch.Tensor) -> torch.Tensor:
    ...
def my_reduce_scatter(tensor: torch.Tensor) -> torch.Tensor:
    ...
def my_all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    ...
```

**General constraints (very important):**

1. For **most collectives**, you **must not** call any of the following in your final submitted code:
   - `dist.broadcast`, `dist.all_reduce`, `dist.reduce`, `dist.all_gather`,
     `dist.gather`, `dist.scatter`, `dist.reduce_scatter`, `dist.barrier`, etc.
2. You **may only** use:
   - `dist.send`, `dist.recv`
   - rank/world_size queries (`dist.get_rank()`, `dist.get_world_size()`)
   - tensor operations on CPU (`torch` on CPU only)
4. Your functions must be **collective**:
   - They should be called on **all ranks** with compatible tensor shapes/dtypes.
   - They should **block** until all necessary communication is complete.
5. All collectives should be **in-place** on the input tensor (i.e. modify the argument) and also return the final tensor for convenience.

#### `my_broadcast(tensor, src)`

Broadcast the `tensor` from rank `src` to all other ranks.

- Input:
  - On rank `src`: `tensor` contains the source data.
  - On other ranks: `tensor` is allocated with the correct shape and dtype; contents are ignored.
- Output:
  - On **all** ranks: `tensor` should contain the same data as the original source tensor.

A simple (non-optimal) algorithm:

- If `rank == src`: send `tensor` to every other rank.
- Else: receive `tensor` from `src`.

You are free to implement a more optimized algorithm (e.g. binary tree) for bonus marks.

![Broadcast Diagram](resources/broadcast.png)

#### `my_allreduce(tensor)` (sum)

Perform an **allreduce with sum** over all ranks.

- Let `x_r` be the input tensor on rank `r`.
- After the operation, **all ranks** should hold `sum_r x_r`.

A simple (non-optimal) algorithm:

1. **Gather to root**
   - Each non-root rank sends its tensor to a root rank (e.g. 0).
   - The root rank receives and accumulates (`+=`) the tensors.
2. **Broadcast the result**
   - Root then broadcasts the final sum tensor to all ranks using your `my_broadcast`.

This would be `O(world_size)` messages to/from root. More optimized algorithms (e.g. ring, tree) are welcome for bonus marks, but not required.

![Allreduce Diagram](resources/allreduce.png)

#### `my_allgather(tensor)`

Allgather concatenates tensors from all ranks along the **first dimension**.

- Input:
  - On rank `r`, `tensor_r` has shape `[N, ...]` (same `N` on all ranks).
- Output:
  - On **each rank**, the returned tensor should have shape `[world_size * N, ...]` containing tensors from ranks in rank order: `[tensor_0; tensor_1; ...; tensor_{world_size-1}]`.

A simple algorithm:

- Pick a root rank (e.g. 0).
- All non-root ranks send their tensor to root.
- Root creates a big buffer, copies all tensors into it in order.
- Root then sends the big buffer to each non-root rank.
- Non-root ranks receive the big buffer and return it.

Again, more optimized implementations can earn bonus marks.

![Allgather Diagram](resources/allgather.png)

#### `my_reduce_scatter(tensor)` (sum)

Reduce-scatter performs an element-wise reduction (sum) and scatters the result.

- Input:
  - On rank `r`, `tensor_r` has shape `[world_size * N, ...]` (same `N` on all ranks).
  - The input tensor is conceptually divided into `world_size` equal chunks along dimension 0.
- Output:
  - Chunk `i` from all ranks are summed element-wise.
  - Rank `i` receives the summed result of chunk `i`.
  - On rank `r`, the returned tensor has shape `[N, ...]` and contains the sum of chunk `r` from all ranks.

A simple algorithm:

- Each rank splits its input tensor into `world_size` chunks.
- For each rank pair `(r, i)`:
  - Rank `r` sends its chunk `i` to rank `i`.
  - Rank `r` receives chunk `r` from rank `i`.
  - Rank `r` accumulates the received chunk into its output buffer.
- To avoid deadlock, use a consistent send/recv ordering (e.g., lower-ranked process sends first).

![Reduce-Scatter Diagram](resources/reducescatter.png)

#### `my_all_to_all(tensor, scatter_dim, gather_dim)`

All-to-all with flexible scatter and gather dimensions. This is used for Ulysses-style sequence parallelism where we need to redistribute data across different dimensions.

- Input:
  - `tensor`: Input tensor to redistribute
  - `scatter_dim`: Dimension to split and scatter across ranks (must be divisible by world_size)
  - `gather_dim`: Dimension to gather received chunks along
- Output:
  - The input is split into `world_size` chunks along `scatter_dim`
  - Rank `r` sends chunk `i` (along scatter_dim) to rank `i`
  - Rank `r` receives chunk `r` (along scatter_dim) from each rank and gathers them along `gather_dim`
  - When `scatter_dim != gather_dim`: scatter_dim shrinks by factor of world_size, gather_dim grows by factor of world_size
  - When `scatter_dim == gather_dim`: output shape equals input shape (redistribution only)

Example:
- Input on each rank: `[batch, seq_len/N, n_heads, head_dim]` where N=world_size
- `scatter_dim=2` (heads), `gather_dim=1` (sequence)
- Output on each rank: `[batch, seq_len, n_heads/N, head_dim]`

Algorithm:

- Split input tensor into `world_size` chunks along `scatter_dim`
- For each other rank `i`:
  - Send chunk `i` (from scatter_dim split) to rank `i`
  - Receive from rank `i` into output chunk `i` (along gather_dim)
- Copy local chunk directly without communication
- Use non-blocking send/recv (isend/irecv) to avoid deadlock

This is essential for redistributing data across ranks in sequence parallelism and tensor parallelism scenarios.

![All-to-All Diagram](resources/alltoall.png)

### Running the Task 1 Unit Tests

From the root directory:

```bash
pytest -q tests/test_broadcast.py tests/test_allreduce.py tests/test_allgather.py tests/test_reduce_scatter.py tests/test_all_to_all.py
```

The tests will internally spawn multiple processes (using `torch.multiprocessing.spawn`) and initialize a small Gloo process group using CPU only. They will call your implementations and check correctness.

You should see all Task 1 tests passing before moving on.

### Schematic Diagram (draw.io recommended)

**All-to-All diagram (5 marks)**

For `my_all_to_all` with `world_size = 4`, draw a schematic showing:

1. How each rank's input tensor is split into chunks.
2. Which chunks are sent from each rank to which destination ranks.
3. How the received chunks are assembled at each rank.

Draw a time-vs-rank diagram where:
- One axis shows ranks 0, 1, 2, 3.
- The other axis shows time/message order.
- Clearly indicate which data chunks are being transferred between ranks with labeled arrows.

You may use draw.io or any preferred tool, as long as the final figure is readable.

Export the diagram as **PDF** or **PNG** and include it in your submission as
`all_to_all_schematic.pdf` or `all_to_all_schematic.png`.

### Written Questions (Task 1)

Answer the following questions in a separate file `written_answers_task1.md` or `written_answers_task1.pdf`.

#### Q1. Broadcast values (5 marks)

Suppose `world_size = 4` and `src = 1`. Each rank `r` initializes:

```python
x = torch.tensor([10 * (r + 1)], dtype=torch.float32)
y = my_broadcast(x, src=1)
print(f"rank {r}: x={x.item()}, y={y.item()}")
```

Assume your broadcast is correct (semantically), regardless of its internal algorithm.

1. What will be printed on each rank (0, 1, 2, 3)? Fill in a table:

| rank | printed `x` | printed `y` |
|------|-------------|-------------|
| 0    |             |             |
| 1    |             |             |
| 2    |             |             |
| 3    |             |             |

Briefly explain why.

---

#### Q2. Allreduce intermediate state (5 marks)

Consider the **simple two-phase allreduce algorithm** described in this handout (gather to root 0 then broadcast).

Let `world_size = 4`, and each rank initializes:

```python
x = torch.tensor([r + 1], dtype=torch.float32)  # 1 on rank 0, 2 on rank 1, etc.
```

Assume the gather phase proceeds in the following order of `recv` on root 0:

1. Rank 1 sends, root receives and accumulates.
2. Rank 2 sends, root receives and accumulates.
3. Rank 3 sends, root receives and accumulates.

1. What is the value of the accumulation tensor on root 0 after each receive?
   - After receiving from rank 1:
   - After receiving from rank 2:
   - After receiving from rank 3:
2. What is the final value printed on **each rank** after the broadcast phase completes?

Write down your reasoning.

---

#### Q3. Allgather layout (5 marks)

Let `world_size = 3` and each rank holds:

```python
# on rank r
x = torch.tensor([r, r + 10], dtype=torch.float32)  # shape [2]
y = my_allgather(x)
```

1. What is the shape of `y` on each rank?
2. What is the **exact content** of `y` on each rank? (Write it as a 1-D array.)

Explain briefly.

---

### Marking Scheme (Task 1 — 50 marks)

- **Implementation correctness (30 marks)**
  - All collective communication functions pass the provided unit tests.
  - Correct use of only `send`/`recv` primitives (no built-in collectives).
  - Proper handling of different world sizes and tensor shapes.
- **All-to-All schematic diagram (5 marks)**
  - Clear visualization of data flow between ranks.
  - Correct representation of chunk splitting and reassembly.
- **Written questions (15 marks)**
  - Q1: Broadcast values (5 marks)
  - Q2: Allreduce intermediate state (5 marks)
  - Q3: Allgather layout (5 marks)

---

## Task 2 (50 marks) — Transformer Block with Tensor Parallel & Ulysses Sequence Parallel

In this task, you are given a simple **Transformer encoder block** implemented in `src/transformer.py`. 

The block operates on inputs of shape:

```text
[batch, seq_len, d_model]
```

### Provided Code

Open `src/transformer.py`. You will find:

- `BasicTransformerBlock` — a **reference implementation** that:
  - Does **not** use `torch.distributed` at all.
  - Works on a single process, CPU-only.
- Three **stub methods** which you must implement:

```python
class BasicTransformerBlock(nn.Module):
    ...

    def init_tensor_parallel_shards(self):
        """
        TODO: Initialize sharded weights for tensor parallelism.
        Each rank extracts its portion of the model weights.
        """
        raise NotImplementedError

    def forward_tensor_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement tensor-parallel forward pass using the sharded weights.
        """
        raise NotImplementedError

    def forward_ulysses(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement Ulysses-style sequence-parallel forward pass.
        Uses world_size for determining parallelism.
        """
        raise NotImplementedError
```

The **non-parallel** forward path is already implemented:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # standard Transformer encoder block on a single process
    ...
```

### Tensor Parallelism Requirements

You must implement **tensor parallelism over the model dimension** following the Megatron-LM approach [1]. For this assignment, we simplify the requirements:

- The process group is the one initialized by `dist_init.init_process_group(backend="gloo")`.
- Use `dist.get_rank()` and `dist.get_world_size()` to obtain the model-parallel rank and world size.
- We assume **model parallel group = the whole world** (no data-parallel split).
- Each rank is responsible for a **shard** of the hidden dimension `d_model` (you may assume `d_model` is divisible by `world_size`).

**Goal:**

You must implement **two methods**:

1. **`init_tensor_parallel_shards()`**: This method initializes the sharded weights.
   - Each rank should extract its portion of the full model weights.
   - Store the sharded weights in the following attributes:
     - `self.q_proj_shard`: Shard of Q projection weight (column-parallel)
     - `self.k_proj_shard`: Shard of K projection weight (column-parallel)
     - `self.v_proj_shard`: Shard of V projection weight (column-parallel)
     - `self.o_proj_shard`: Shard of O projection weight (row-parallel)
     - `self.fc1_shard`: Shard of FC1 weight (column-parallel)
     - `self.fc1_bias_shard`: Shard of FC1 bias (if present)
     - `self.fc2_shard`: Shard of FC2 weight (row-parallel)
   - **Column-parallel** means sharding along the output dimension (rows of weight matrix).
   - **Row-parallel** means sharding along the input dimension (columns of weight matrix).

2. **`forward_tensor_parallel(x)`**: This method performs the forward pass using the sharded weights.
   - Assume `init_tensor_parallel_shards()` has been called first.
   - Use the sharded weights stored in `self.*_shard` attributes.
   - Must return on **every rank** the **same result tensor** as `forward(x)` would produce on a single process.
   - You should use **standard PyTorch distributed collectives** for communication:
     - `dist.all_reduce()` for summing partial results
     - `dist.all_gather()` for gathering shards
   - You may also use your custom collectives from Task 1 if you prefer, but standard collectives are recommended.

The unit tests will check **numerical equivalence** between:

- `y_ref = block.forward(x)` (single-process semantics)
- After calling `block.init_tensor_parallel_shards()`, `y_tp = block.forward_tensor_parallel(x)` (multi-process, with Gloo+CPU), run on each rank.

### Ulysses-Style Sequence Parallel Requirements

You must implement a simplified version of **Ulysses sequence parallelism** [2], where the **sequence dimension** is partitioned across ranks.

We consider the following simplified setting:

- Input `x_shard` has shape `[batch, seq_len/world_size, d_model]`.
- The parallelism is determined by `world_size` (the number of distributed processes).
- Self-attention requires **global attention** (tokens can attend to all positions). Therefore, ranks need to exchange information about the keys and values.

Your `forward_ulysses(x_shard)` must:

1. Produce the **same result** as `forward(x)` (reference block), on every rank.
2. Use **your custom `my_all_to_all()` function from Task 1** for communication:
   - You **must** use `my_all_to_all()` (from `src/collectives.py`) for exchanging sequence chunks
   - This is required because the Gloo backend does not support `dist.all_to_all()`
   - You may also use `dist.all_gather()` or `dist.all_reduce()` if needed for your approach
3. The input `x_shard` to `forward_ulysses` is a **shard** of the full sequence (shape `[batch, local_seq_len, d_model]`), not the full sequence.
4. The output `x_shard` should also be a **shard** (shape `[batch, local_seq_len, d_model]`).

### Running the Task 2 Unit Tests

The Task 2 tests live in `tests/test_transformer.py`.

You can run them with:

```bash
pytest -q tests/test_transformer.py
```

These tests will:

- Spawn multiple processes using `run_distributed` from `tests/utils.py`.
- In each worker:
  - Initialize a process group (`gloo`, CPU).
  - Create a `BasicTransformerBlock` with **identical weights on all ranks** via a fixed random seed.
  - Construct a random input batch `x`.
  - Compute:
    - `y_ref = block.forward(x)` (single-process semantics)
    - `y_tp = block.forward_tensor_parallel(x)`
    - `y_ulysses = block.forward_ulysses(x_shard)`
  - Check that:
    - `y_tp` matches `y_ref` (within a tolerance).
    - `y_ulysses` matches `y_ref`.

Until you implement the two parallel methods, these tests will fail with `NotImplementedError`.

### Marking Scheme (Task 2 — 50 marks)

- **Correctness of tensor-parallel forward (20 marks)**
  - `forward_tensor_parallel` returns numerically close results to `forward`.
  - Works for different world sizes used in tests.
- **Correctness of Ulysses-style sequence-parallel forward (20 marks)**
  - `forward_ulysses` returns numerically close results to `forward`.
  - Uses sequence partitioning and communication appropriately.
- **Use of custom collectives / communication design (10 marks)**
  - Reasonable use of `send`/`recv`-based patterns.
  - Use of Task 1 collectives where appropriate (`my_allgather`, `my_allreduce`).
  - No built-in collectives used.

You may optionally include **short written notes** in your submission describing your parallelization strategy (not graded but helpful for partial credit in code review).

---

## Overall Submission Checklist

You should submit at least:

1. **Code**
   - `src/collectives.py` — Task 1 collectives.
   - `src/transformer.py` — Task 2 parallel Transformer methods.
2. **Diagram**
   - `all_to_all_schematic.(pdf|png)` — All-to-All diagram (Task 1).
3. **Written answers**
   - `written_answers_task1.(md|pdf)` — Task 1 Q1–Q3 answers.

Make sure that:

- Your code runs on CPU only.
- You are using the `gloo` backend.
- You do **not** use built-in collectives in your task1 solutions (except for your own sanity checks in separate, non-submitted scripts).
- `pytest` passes on the instructor's environment (after you finish all tasks).

---

## References

[1] Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.** arXiv preprint arXiv:1909.08053.

[2] Jacobs, S. A., Tanaka, M., Zhang, C., Zhang, M., Song, L., Rajbhandari, S., & He, Y. (2023). **DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models.** arXiv preprint arXiv:2309.14509.


---
layout: single
title:  "Profile CUDA UVM Performance"
date:   2024-03-25
author_profile: true
comments: true
tags: [CUDA, UVM, Profiler]
---

This post is to log my profile of CUDA unified virtual memory.

# Summary of Issues

1. Extremely slow on manually swap when run `export CUDA_VISIBLE_DEVICES=X` at the beginning. [Performance drop after specifying CUDA_VISIBLE_DEVICES=0 - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/performance-drop-after-specifying-cuda-visible-devices-0/288557), solved by pinned core or pinned memory

2. Compile with `-O3` will also effect the performance of baseline, and set visible devices won't affect it much then.

# Virtual Address Fragmentation

## Setup

Running on A100 with 80GB memory, CUDA 12.2, driver 535.161.07.

Assume `k` requests coming in, every request has a individual virtual memory space for KV cache `hidden_size x max_embeddings`. 

Every cycle randomly selects a group of request and generate a number of new tokens to emulate generation progress. Then, randomly select a group of request to free and add new to simulate query out and in.

Compare the theoretical space (sum of KV cache size) and allocated space (GPU memory utilization).

## Results

# Oversubscription Performance

## Setup

Running on A100 with 80GB memory, CUDA 12.2, driver 535.161.07.

Split GPU memory into `k` blocks and allocate `m` chunks of data. Chunk size is equation to one partition, and `m > k`. 

For example, if we launch with `(k, m) = (4, 8)`, each chunk takes about 20GB memory and we need to traverse through 8 chunks in total.

The test includes three phases, 

1. Prefill: The first `k` blocks will fills up the GPU memory, which only needs HtoD memory copy without evicting blocks.

2. First cycle: For the next `m-k` chunks, the system also need to evict one of the resident chunk to host by DtoH copy, then copy the new chunk to the empty block.

3. Second cycle: Now all the blocks are filled with `k`chunks, we will issue the second cycle that iteratively load `m` chunks. Every iteration includes sequential DtoH and HtoD copy.

## Experiements

### Baseline: Manually swapping

wierd output when export visible device

| m    | k    | prefill [s] | 1st cycle[s] | 2nd cycle | total [s] |
|:----:|:----:| ----------- | ------------ | --------- | --------- |
| 8    | 4    | 19.20       | 36.81        | 27.42     | 64.23     |
| 12   | 4    | 19.32       | 57.96        | 40.92     | 98.89     |
| 80   | 40   | 19.31       | 43.33        | 35.63     | 78.97     |
| 120  | 40   | 19.58       | 59.41        | 41.02     | 100.42    |
| 2048 | 1024 | 19.64       | 37.05        | 27.90     | 64.95     |
| 3072 | 1024 | 19.82       | 59.08        | 41.42     | 100.50    |

### Manually swapping with pinned host

wierd output when export visible device

| m    | k    | prefill [s] | 1st cycle[s] | 2nd cycle | total [s] |
|:----:|:----:| ----------- | ------------ | --------- | --------- |
| 8    | 4    | 3.86306     | 11.5981      |           |           |
| 12   | 4    |             |              |           |           |
| 80   | 40   |             |              |           |           |
| 120  | 40   |             |              |           |           |
| 2048 | 1024 |             |              |           |           |
| 3072 | 1024 |             |              |           |           |

### Using UVM

#### Naive

| m    | k    | prefill [s] | 1st cycle[s] | 2nd cycle | total [s] |
|:----:|:----:| ----------- | ------------ | --------- | --------- |
| 8    | 4    | 21.06       | 42.64        | 44.43     | 87.07     |
| 12   | 4    | 21.26       | 67.43        | 66.86     | 134.28    |
| 80   | 40   | 17.13       | 36.26        | 37.98     | 74.24     |
| 120  | 40   | 16.97       | 54.70        | 56.91     | 111.62    |
| 2048 | 1024 | 20.59       | 43.44        | 47.37     | 90.81     |
| 3072 | 1024 | 20.77       | 66.66        | 70.62     | 137.28    |

#### Prefetch

| m    | k    | prefill [s] | 1st cycle[s] | 2nd cycle | total [s] |
|:----:|:----:| ----------- | ------------ | --------- | --------- |
| 8    | 4    | 3.48        | 10.31        | 18.01     | 28.32     |
| 12   | 4    | 3.48        | 17.15        | 26.03     | 43.17     |
| 80   | 40   | 3.54        | 10.80        | 18.16     | 28.96     |
| 120  | 40   | 3.54        | 18.07        | 26.28     | 44.35     |
| 2048 | 1024 | 3.57        | 11.14        | 18.72     | 29.86     |
| 3072 | 1024 | 3.57        | 18.74        | 27.09     | 45.83     |

#### Advise [TODO]

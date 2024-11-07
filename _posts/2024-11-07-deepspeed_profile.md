---
layout: single
title:  "Use Nsight System to Profile a Model Training with DeepSpeed on Multi-Node Cluster"
date:   2024-11-07
author_profile: true
comments: true
tags: [Deepspeed, Training, Nsight, Multi-GPU, Cluster, Multi-Node]
---

This post is to log how I managed to profile a model training running on multiple nodes in a cluster with DeepSpeed and Nsight System. Click [here](#final-implementation) to jump to the final implementation code.

## Introduction
Well DeepSpeed and Nsight System are both powerful tools. It is very simple and smooth to profile a single-node multi-GPU training process. However, when it comes to multi-node multi-GPU training, things get complicated. Especially when the cluster is managed by SLURM. Thus I spent several hours to figure out how to profile the training process.

## The Problem

I started with the following script, which is a normal DeepSpeed launch command:
```bash
# NOTE(liuxs): don't enable both asymmetric ep and extended expert
deepspeed --master_port 12345 \
          --hostfile=${HOSTFILE} \
          --bind_cores_to_rank \
        train.py --deepspeed_config=ds_config.json \
        --steps 3 \
        ...
```

I run it under `nsys profile` command, and found out the program stuck at this line:
```bash
[2024-11-07 14:42:00,299] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
```

When I click `Ctrl+C` to interrupt the program, I got the following error traceback:
```bash
Traceback (most recent call last):
  File "<path_to_deepspeed_env>/bin/deepspeed", line 7, in <module>
    exec(compile(f.read(), __file__, 'exec'))
  File "<path_to_workspace>/bin/deepspeed", line 6, in <module>
    main()
  File "<path_to_workspace>/deepspeed/launcher/runner.py", line 484, in main
    result = subprocess.check_output(hostname_cmd)
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 421, in check_output
    return run(*popenargs, stdout=PIPE, timeout=timeout, check=True,
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 505, in run
    stdout, stderr = process.communicate(input, timeout=timeout)
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 1146, in communicate
    self.wait()
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 1209, in wait
    return self._wait(timeout=timeout)
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 1959, in _wait
    (pid, sts) = self._try_wait(0)
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 1917, in _try_wait
    (pid, sts) = os.waitpid(self.pid, wait_flags)
KeyboardInterrupt
```
I dug into the DeepSpeed source code and found out the root cause is that the program is stuck at this line:
```python
    if not args.master_addr:
        assert multi_node_exec
        first_host = list(active_resources.keys())[0]
        ssh_check_cmd = "ssh "
        if args.ssh_port is not None:
            ssh_check_cmd += f" -p {args.ssh_port}"
        ssh_check_cmd += f" {first_host} hostname -I"
        hostname_cmd = shlex.split(ssh_check_cmd)
        try:
            result = subprocess.check_output(hostname_cmd) # where it stuck
        except subprocess.CalledProcessError as err:
            logger.error(
                "Unable to detect suitable master address via `hostname -I`, please manually specify one via --master_addr"
            )
            raise err
```
This indicates that the program is waiting for getting the master address via `hostname -I`. Let's skip it by providing the master address manually:
```bash
# NOTE(liuxs): don't enable both asymmetric ep and extended expert
deepspeed --master_addr ${MASTER_ADDR} --master_port 12345 \
          --hostfile=${HOSTFILE} \
          --bind_cores_to_rank \
        $1 --deepspeed_config=ds_config.json \
        --steps 3 \
```

However, the program is stuck at this line:
```bash
Traceback (most recent call last):
  File "<path_to_deepspeed_env>/bin/deepspeed", line 7, in <module>
    exec(compile(f.read(), __file__, 'exec'))
  File "<path_to_workspace>/bin/deepspeed", line 6, in <module>
    main()
  File "<path_to_workspace>/deepspeed/launcher/runner.py", line 469, in main
    subprocess.check_call(safe_ssh_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 364, in check_call
    retcode = call(*popenargs, **kwargs)
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 347, in call
    return p.wait(timeout=timeout)
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 1209, in wait
    return self._wait(timeout=timeout)
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 1959, in _wait
    (pid, sts) = self._try_wait(0)
  File "<path_to_deepspeed_env>/lib/python3.10/subprocess.py", line 1917, in _try_wait
    (pid, sts) = os.waitpid(self.pid, wait_flags)
KeyboardInterrupt
```

Looks like it fails on checking if the passwordless-ssh is working properly with this hostfile:
```python
    # validate that passwordless-ssh is workly properly with this hostfile
    if multi_node_exec and not args.no_ssh_check and not args.no_ssh:
        first_host = list(active_resources.keys())[0]
        try:
            ssh_check_cmd = "ssh -o PasswordAuthentication=no "
            if args.ssh_port is not None:
                ssh_check_cmd += f"-p {args.ssh_port} "
            ssh_check_cmd += f"{first_host} hostname"
            safe_ssh_cmd = shlex.split(ssh_check_cmd)
            subprocess.check_call(safe_ssh_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) # where is stuck
        except subprocess.CalledProcessError:
            raise RuntimeError(
                f"Using hostfile at {args.hostfile} but host={first_host} was not reachable via ssh. If you are running with a single node please remove {args.hostfile} or setup passwordless ssh."
            )
```
We can easily skip it by setting the `--no_ssh_check` option, which is like,
```bash
# NOTE(liuxs): don't enable both asymmetric ep and extended expert
deepspeed --master_addr ${MASTER_ADDR} --master_port 12345 \
          --hostfile=${HOSTFILE} --no_ssh_check\
          --bind_cores_to_rank \
        $1 --deepspeed_config=ds_config.json \
        --steps 3 \
```

Now the program is running on both nodes, but when it finishes, I stuck at this line:
```bash
gl1517: [2024-11-07 14:50:49,659] [INFO] [launch.py:348:main] Process 1947770 exits successfully.
gl1021: [2024-11-07 14:50:49,709] [INFO] [launch.py:348:main] Process 3219950 exits successfully.
# stop right here
```
And the nsight system is not generating report as usual. 
Till now, I suppose there is no actual way to profile two nodes connected through PDSH under one Nsight System process as they are not on the same machine. Another option is to run the training script on each node separately with deepspeed under Nsight System.

Refer to the guide of deepspeed [here](https://www.deepspeed.ai/getting-started/#launching-without-passwordless-ssh), they've supported the `--no_ssh` option that you can manually launch deepspeed on each node and they will sync up automatically. However, from my testing, the `--no_ssh` option is not supported in the latest version (0.15.x I suppose). Therefore, my final solution is below.

## Final Implementation

First of all, upgrade the DeepSpeed to the latest version. Make sure the option `--no_ssh` is supported in the `deepspeed/launcher/runner.py`. If you are not able to upgrade, you can manually copy and paste the lastest [runner.py](https:/github.com/microsoft/DeepSpeed/blob/master/deepspeed/launcher/runner.py) to your code.

Then create a `hostfile`, for example:
```bash
gl1500 slots=1
gl1000 slots=1
```
Normally the first line is the master node.

Then you can launch the training script with DeepSpeed as follows:
```bash
deepspeed --master_addr gl1500 --master_port 12345 \
          --hostfile=hostfile --no_ssh --node_rank <GLOBAL_RANK> \
          --bind_core_list <BIND_CORE_LIST>  \
        <train.py> --deepspeed_config=ds_config.json \
        --steps 3 \
        ...
```
On the master node (gl1500), `GLOBAL_RANK` is `0`, while on gl1000, it is `1`.

The `BIND_CORE_LIST` is a list of core indices assigned to the current node. If your cluster is managed by SLURM, you can get the list by `taskset -cp $$`. You can run the following command to get the core list in required format:
```bash
BIND_CORE_LIST=$(taskset -cp $$ | awk -F: '{print $2}' | tr -d '[:space:]')
# example return: 7,16-23
```

You can write them into a shell script and run the script with `bash`. For example, mine is like,
```bash
#!/bin/bash

# bash run.sh <train.py> <global_rank> <master_addr>
# host file specifying the number of nodes and GPUs
HOSTFILE=hostfile

SEQ_LEN=1024

MASTER_ADDR=$3
# read code binding from taskset -cp $$
BIND_CORE_LIST=$(taskset -cp $$ | awk -F: '{print $2}' | tr -d '[:space:]')

# NOTE(liuxs): don't enable both asymmetric ep and extended expert
deepspeed --master_addr ${MASTER_ADDR} --master_port 12345 \
          --hostfile=${HOSTFILE} --no_ssh --node_rank=${2} \
          --bind_core_list ${BIND_CORE_LIST} \
        $1 --deepspeed_config=ds_config.json \
        --steps 3 \
        --seq_len ${SEQ_LEN} \
        ...
```
Now if you want to profile the training process with Nsight System, you can run the following command:
```bash
# on the master node (gl1500)
nsys profile -o nsys_log/nsys_report_rank0 --force-overwrite true --trace cuda,nvtx --cuda-memory-usage true bash run.sh train.py 0 gl1500
```

On the worker node (gl1000), you can run the following command:
```bash
# on the worker node (gl1000)
nsys profile -o nsys_log/nsys_report_rank1 --force-overwrite true --trace cuda,nvtx --cuda-memory-usage true bash run.sh train.py 1 gl1500
```

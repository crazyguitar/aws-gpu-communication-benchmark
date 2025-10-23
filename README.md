# GPU Communication Over AWS EFA Benchmarking

The efficiency of GPU communication and interconnects is a key determinant of
large language model (LLM) inference/training performance, complementing computational
capabilities. In this study, we conduct a systematic benchmark of two widely
adopted communication libraries, [NCCL](https://github.com/NVIDIA/nccl) and
[NVSHMEM](https://github.com/NVIDIA/nvshmem),  on the AWS Elastic Fabric Adapter
(EFA) interconnect. Our evaluation focuses on collective communication
operations, such as all-to-all, which are known to pose significant scalability
challenges in Mixture-of-Experts (MoE) architectures. The [data](data/) directory
contains the raw benchmarking results, which serve as the basis for all figures
and analyses presented in the README.

## Launch benchmarking jobs on K8S

All experiments were conducted on [Amazon SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks.html)
using `p5.48xlarge` instances, each configured with four Elastic Fabric Adapter
(EFA) interfaces per GPU. The Kubernetes manifests used in this study are
available in the [k8s](k8s/) directory. The experiments were executed using the
commands described below:

```bash
# build a docker image on your local machine
docker build -t gpucomm:latest .

# Launch a benchmark job on a Kubernetes cluster
kubectl -f  k8s/<job>.yaml
```

## Launch benchmarking jobs on Slurm

Although our experiments were conducted on a Kubernetes cluster, we also
provide an example demonstrating how to launch a job on a
[Slurm](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-slurm.html)
cluster using enroot. The commands provided below illustrate steps for submitting
an example job.

```bash
# Login to a compute node
srun -N 1 --pty /bin/ash

# Build a enroot sqush file
make sqush

# Start an enroot environment and build binaries
enroot create --name gpucomm gpucomm+latest.sqsh
enroot start --mount /fsx:/fsx gpucomm /bin/bash
make

# Launch a NVSHMEM benchmark job on a Slurm cluster
salloc -N 2
bash slurm/jacobi_nvshmem.sh
```

## Evaluation

The following figure illustrates the relationship between the number of GPUs and
the achieved bandwidth of NCCL, measured using [nccl-tests](https://github.com/NVIDIA/nccl-tests).
The results show that the bandwidth of the All-to-All collective communication is
significantly lower compared to that of All-Gather and All-Reduce operations.
Furthermore, as the number of GPUs increases, both performance and bandwidth
exhibit noticeable degradation, potentially impacting overall training and inference
efficiency. For instance, in our experiments, the All-to-All bandwidth decreases
to approximately 40 GB/s, which may become a performance bottleneck for
Mixture-of-Experts (MoE) layers when aggregating computation results.

![alt NCCL](imgs/nccl_bandwidth.png)

[DeepEP](https://github.com/deepseek-ai/DeepEP) leverages NVSHMEM to implement Mixture-of-Experts (MoE) layers with
low-latency, GPU-initiated communication. Our experimental results indicate that
when the number of GPUs is below 32, NCCL achieves higher performance in All-to-All
operations. However, an interesting observation is that the performance degradation
of NVSHMEM remains minimal as the GPU count increases, suggesting that it may
offer better scalability for MoE layers with a large number of experts.

![alt NCCL_NVSHMEM_alltoall](imgs/alltoall_comparison.png)

AWS EFA supports GPU-initiated communication but does not provide capabilities
equivalent to InfiniBand GPUDirect Async (IBGDA). Instead, it employs a mechanism
similar to InfiniBand Reliable Connection (IBRC), which relies on a proxy buffer
to handle GPU-initiated communication. Based on our investigation, we observed
that GPU-initiated communication exhibits higher latency compared to CPU-initiated
communication, as measured by the [alltoall\_latency.cu](https://github.com/NVIDIA/nvshmem/blob/devel/perftest/device/coll/alltoall_latency.cu)
performance test using NVSHMEM’s blocking calls. However, these results do not
imply that GPU-initiated communication over EFA is inherently slow. Further
details and analysis will be provided in the Jacobi experiments.

![alt NVSHMEM_alltoall](imgs/nvshmem_comparison.png)

Before examining CPU- and GPU-initiated communication, we observed that
NVSHMEM throughput is highly dependent on packet size. As shown in the following
figures, excessively large packet sizes lead to a significant degradation in
All-to-All performance. Interestingly, this performance degradation appears to
be largely insensitive to the number of GPUs.

![alt NVSHMEM_alltoall_size](imgs/nvshmem_size_comparison.png)

The project [Multi GPU Programming Models](https://github.com/NVIDIA/multi-gpu-programming-models)
provides a step-by-step guide for comparing the performance of NCCL and NVSHMEM
using the Jacobi algorithm in multi-GPU environments. In our experiments, we
observed that the baseline NCCL implementation achieves better performance
compared to variants employing Overlap, CUDA Graphs, or User Buffers.
Although these additional features are theoretically expected to improve
performance, they did not demonstrate measurable benefits in our tests.

Another key observation is that NVSHMEM with blocking GPU-initiated communication
exhibits lower performance relative to other implementations. However, when
non-blocking GPU-initiated communication is used with appropriate computation
overlapping and synchronization, NVSHMEM achieves the best performance, both
in terms of throughput and efficiency. These results indicate that fully
leveraging NVSHMEM requires careful algorithm design, emphasizing proper overlap
between communication and computation as well as effective synchronization.

![alt Jacobi](imgs/jacobi.png)

## Reference

* [Improving Network Performance of HPC Systems Using NVIDIA Magnum IO NVSHMEM and GPUDirect Async](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)
* [NVIDIA/multi-gpu-programming-models](https://github.com/NVIDIA/multi-gpu-programming-models)
* [ISC25 Tutorial: Efficient Distributed GPU Programming for Exascale](https://github.com/FZJ-JSC/tutorial-multi-gpu)

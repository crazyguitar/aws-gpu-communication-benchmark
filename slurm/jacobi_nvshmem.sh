#!/bin/bash

set -exo pipefail

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
sqsh="${DIR}/../gpucomm+latest.sqsh"
mount="/fsx:/fsx"
binary="${DIR}/../build/third_party/multi-gpu-programming-models/nvshmem/nvshmem"

cmd="$(cat <<EOF
export NCCL_DEBUG_SUBSYS=off
export NCCL_DEBUG=INFO
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-ofi-tuner.so
export NCCL_NVLS_ENABLE=0
export NVSHMEM_DEBUG=INFO
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
${binary}
EOF
)"

srun --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name nvshmem \
  --mpi=pmix \
  --ntasks-per-node=8 \
  bash -c "${cmd}"

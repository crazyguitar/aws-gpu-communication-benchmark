#!/bin/bash

set -exo pipefail

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
sqsh="${DIR}/../gpucomm+latest.sqsh"
mount="/fsx:/fsx"
binary="${DIR}/../build/third_party/multi-gpu-programming-models/nccl_overlap/nccl-overlap"

cmd="$(cat <<EOF
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export NCCL_DEBUG_SUBSYS=off
export NCCL_DEBUG=INFO
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-ofi-tuner.so
export NCCL_NVLS_ENABLE=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=524288
${binary} -symmetric_memory_reg
EOF
)"

srun --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name nccl \
  --mpi=pmix \
  --ntasks-per-node=8 \
  bash -c "${cmd}"

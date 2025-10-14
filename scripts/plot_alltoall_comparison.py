import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read data files
nccl_alltoall = pd.read_csv('../data/nccl_alltoall.csv')
nvshmem_host = pd.read_csv('../data/nvshmem_alltoall_host.csv')

# Filter for specific message sizes (32MB, 64MB, 128MB)
target_sizes = [32, 64, 128]
nccl_filtered = nccl_alltoall[nccl_alltoall['size(MB)'].isin(target_sizes)]
nvshmem_filtered = nvshmem_host[nvshmem_host['size(MB)'].isin(target_sizes)]

# Find intersection of GPU counts for filtered data
nccl_gpus = set(nccl_filtered['GPUs'].unique())
nvshmem_gpus = set(nvshmem_filtered['GPUs'].unique())
common_gpus = nccl_gpus.intersection(nvshmem_gpus)

# Filter to common GPU counts
nccl_final = nccl_filtered[nccl_filtered['GPUs'].isin(common_gpus)]
nvshmem_final = nvshmem_filtered[nvshmem_filtered['GPUs'].isin(common_gpus)]

# Add library type and size info
nccl_final['Library'] = 'NCCL'
nvshmem_final['Library'] = 'NVSHMEM Host'

# Use appropriate bandwidth column
nccl_final['Bandwidth'] = nccl_final['out algbw(GB/s)']
nvshmem_final['Bandwidth'] = nvshmem_final['algbw(GB/s)']

# Combine data
df = pd.concat([
    nccl_final[['GPUs', 'size(MB)', 'Bandwidth', 'Library']],
    nvshmem_final[['GPUs', 'size(MB)', 'Bandwidth', 'Library']]
], ignore_index=True)

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, size in enumerate(target_sizes):
    size_data = df[df['size(MB)'] == size]
    sns.lineplot(data=size_data, x='GPUs', y='Bandwidth', hue='Library', marker='o', ax=axes[i], err_style=None)
    axes[i].set_title(f'AllToAll Bandwidth - {size}MB')
    axes[i].set_ylabel('Bandwidth (GB/s)')
    axes[i].legend()

plt.tight_layout()
plt.savefig('../imgs/alltoall_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

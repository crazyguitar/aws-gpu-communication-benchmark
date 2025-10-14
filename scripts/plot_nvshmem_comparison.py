import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read data files
nvshmem_host = pd.read_csv('../data/nvshmem_alltoall_host.csv')
nvshmem_device = pd.read_csv('../data/nvshmem_alltoall_device.csv')

# Convert KB to MB for device data
nvshmem_device['size(MB)'] = nvshmem_device['size(KB)'] / 1024

# Filter for 2MB to 16MB range
host_filtered = nvshmem_host[(nvshmem_host['size(MB)'] >= 2) & (nvshmem_host['size(MB)'] <= 16)]
device_filtered = nvshmem_device[(nvshmem_device['size(MB)'] >= 2) & (nvshmem_device['size(MB)'] <= 16)]

# Add implementation type
host_filtered = host_filtered.copy()
device_filtered = device_filtered.copy()
host_filtered['Implementation'] = 'NVSHMEM Host'
device_filtered['Implementation'] = 'NVSHMEM Device'

# Combine data
df = pd.concat([
    host_filtered[['GPUs', 'size(MB)', 'algbw(GB/s)', 'Implementation']],
    device_filtered[['GPUs', 'size(MB)', 'algbw(GB/s)', 'Implementation']]
], ignore_index=True)

# Get unique sizes for subplots
sizes = sorted(df['size(MB)'].unique())
n_sizes = len(sizes)

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, n_sizes, figsize=(4*n_sizes, 6))

if n_sizes == 1:
    axes = [axes]

for i, size in enumerate(sizes):
    size_data = df[df['size(MB)'] == size]
    sns.lineplot(data=size_data, x='GPUs', y='algbw(GB/s)', hue='Implementation', marker='o', ax=axes[i], err_style=None)
    axes[i].set_title(f'{int(size)}MB')
    axes[i].set_xlabel('Number of GPUs')
    axes[i].set_ylabel('Bandwidth (GB/s)')
    axes[i].legend()

plt.suptitle('NVSHMEM AllToAll: Host vs Device Comparison')
plt.tight_layout()
plt.savefig('../imgs/nvshmem_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

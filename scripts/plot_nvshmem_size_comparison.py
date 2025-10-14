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

# Add implementation type
nvshmem_host['Implementation'] = 'Host'
nvshmem_device['Implementation'] = 'Device'

# Convert GPUs to string for legend
nvshmem_host['GPUs_str'] = nvshmem_host['GPUs'].astype(str) + ' GPUs'
nvshmem_device['GPUs_str'] = nvshmem_device['GPUs'].astype(str) + ' GPUs'

sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Host implementation
sns.lineplot(data=nvshmem_host, x='size(MB)', y='algbw(GB/s)', hue='GPUs_str', marker='o', ax=ax1, err_style=None)
ax1.set_title('NVSHMEM AllToAll - Host')
ax1.set_xlabel('Message Size (MB)')
ax1.set_ylabel('Bandwidth (GB/s)')
ax1.legend()

# Device implementation
sns.lineplot(data=nvshmem_device, x='size(MB)', y='algbw(GB/s)', hue='GPUs_str', marker='o', ax=ax2, err_style=None)
ax2.set_title('NVSHMEM AllToAll - Device')
ax2.set_xlabel('Message Size (MB)')
ax2.set_ylabel('Bandwidth (GB/s)')
ax2.legend()

plt.tight_layout()
plt.savefig('../imgs/nvshmem_size_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Read NCCL data files
all_reduce = pd.read_csv('../data/nccl_all_reduce.csv')
all_gather = pd.read_csv('../data/nccl_all_gather.csv')
alltoall = pd.read_csv('../data/nccl_alltoall.csv')

# Add operation type based on filename
all_reduce['Operation'] = 'All Reduce'
all_gather['Operation'] = 'All Gather'
alltoall['Operation'] = 'AllToAll'

# Combine all data
df_out = pd.concat([all_reduce, all_gather, alltoall], ignore_index=True)
df_in = pd.concat([all_reduce, all_gather, alltoall], ignore_index=True)

# Add suffix for in/out
df_out['Operation'] = df_out['Operation'] + ' (Out)'
df_in['Operation'] = df_in['Operation'] + ' (In)'

sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Out algorithm bandwidth (solid lines)
sns.lineplot(data=df_out, x='GPUs', y='out algbw(GB/s)', hue='Operation', marker='o', ax=ax, err_style=None, linestyle='-')

# In algorithm bandwidth (dashed lines)
sns.lineplot(data=df_in, x='GPUs', y='in algbw(GB/s)', hue='Operation', marker='s', ax=ax, err_style=None, linestyle='--')

ax.set_title('NCCL Algorithm Bandwidth')
ax.set_ylabel('Bandwidth (GB/s)')
ax.legend()

plt.tight_layout()
plt.savefig('../imgs/nccl_bandwidth.png', dpi=300, bbox_inches='tight')
plt.close()

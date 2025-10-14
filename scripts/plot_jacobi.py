import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/jacobi.csv')

name_mapping = {
    'NCCL': 'NCCL',
    'NCCL w/ Overlap': 'NCCL+Overlap',
    'NCCL w/ Overlap + CUDA Graph': 'NCCL+Graph',
    'NCCL w/ Overlap + CUDA Graph + User Buffer': 'NCCL+Buffer',
    'NVSHMEM': 'NVSHMEM',
    'NVSHMEM w/ Block Comm': 'NVSHMEM+Block',
    'NVSHMEM w/ Block Comm + Overlap + Neighborhood Sync': 'NVSHMEM+Opt'
}

df['Short_Name'] = df['Experiment'].map(name_mapping)

sns.set_style("whitegrid")
fig = plt.figure(figsize=(16, 10))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 1, 2)

sns.barplot(data=df, x='GPUs', y='Speedup', hue='Short_Name', ax=ax1)
ax1.set_title('GPU Communication Benchmark - Speedup Comparison')
ax1.get_legend().remove()

sns.barplot(data=df, x='GPUs', y='Efficiency', hue='Short_Name', ax=ax2)
ax2.set_title('GPU Communication Benchmark - Efficiency Comparison')
ax2.set_ylabel('Efficiency (%)')
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=1)

sns.lineplot(data=df, x='GPUs', y='Time(sec)', hue='Short_Name', marker='o', ax=ax3)
ax3.set_title('GPU Communication Benchmark - Latency Comparison')
ax3.set_ylabel('Time (sec)')
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=1)

plt.tight_layout()
plt.savefig('../imgs/jacobi.png', dpi=300, bbox_inches='tight')
plt.show()

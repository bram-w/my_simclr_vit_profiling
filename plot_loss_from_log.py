import numpy as np
import matplotlib.pyplot as plt
import sys
import smart_open

with open(sys.argv[1]) as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines if "loss: 0." in l]

losses = [float(l.split('loss: ')[1].split(',')[0]) for l in lines]

kernel_size = 20

smooth_losses = np.convolve(losses, np.ones(kernel_size)/kernel_size)

plt.plot(smooth_losses[kernel_size:-kernel_size])

with smart_open.open('gs://sfr-tpu-us-east1-research/bwallace/logs/tmp.png', 'wb') as f:
    plt.savefig(f)



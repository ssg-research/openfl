import torch
import argparse
import numpy as np
from PIL import Image
import torchvision as tv


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.transforms import BlendedGenericTransform

import matplotlib
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 1000,  # to adjust notebook inline plot size
    'axes.labelsize': 10, # fontsize for x and y labels (was 10)
    'axes.titlesize': 7,
    'font.size': 10, # was 10
    'legend.fontsize': 7, # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': True,
    'figure.figsize': [3.46, 2.6], #[5.67, 1.05], #[3.46, 2.6], #was 1.2
    'font.family': 'serif',
    'legend.frameon' : False
}

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 10,
        }

matplotlib.rcParams.update(params)

aggregated_test1 = np.asarray([15.0, 96.05, 97.01, 97.64, 97.68, 97.89, 98.02, 98.27, 98.19, 98.22])
aggregated_test2 = np.asarray([17.0, 96.64, 97.58, 97.63, 97.90, 97.88, 98.09, 98.33, 98.37, 98.35])
aggregated_test3 = np.asarray([9.00, 96.85, 97.43, 97.70, 98.19, 98.13, 98.22, 98.36, 98.45, 98.45])
aggregated_test4 = np.asarray([15.0, 96.63, 97.21, 97.61, 97.78, 97.99, 98.07, 98.15, 98.30, 98.42])
aggregated_test5 = np.asarray([8.52, 97.12, 97.40, 97.29, 97.48, 97.77, 97.69, 98.06, 98.28, 98.23])
aggregated_test  = np.vstack((aggregated_test1, aggregated_test2, aggregated_test3, aggregated_test4, aggregated_test5))


aggregated_wm1 = np.asarray([10.0, 30.0, 34.0, 52.0, 74.0, 85.0, 95.0, 99.0, 100.0, 100.0])
aggregated_wm2 = np.asarray([10.0, 19.0, 34.0, 76.0, 80.0, 91.0, 96.0, 99.0, 99.00, 99.00])
aggregated_wm3 = np.asarray([10.0, 30.0, 54.0, 78.0, 78.0, 98.0, 96.0, 99.0, 100.0, 100.0])
aggregated_wm4 = np.asarray([10.0, 24.0, 45.0, 71.0, 83.0, 82.0, 95.0, 94.0, 100.0, 100.0])
aggregated_wm5 = np.asarray([10.0, 34.0, 37.0, 59.0, 87.0, 95.0, 98.0, 99.0, 100.0, 100.0])
aggregated_wm = np.vstack((aggregated_wm1, aggregated_wm2, aggregated_wm3, aggregated_wm4, aggregated_wm5))

clean_test1 = np.asarray( [8.73, 96.58, 97.47, 97.69, 98.03, 98.19, 98.36, 98.52, 98.61, 98.81])
clean_test2 = np.asarray([13.13, 96.73, 97.52, 98.00, 98.34, 98.36, 98.55, 98.64, 98.76, 98.76])
clean_test3 = np.asarray([10.30, 96.71, 97.55, 97.92, 98.13, 98.40, 98.41, 98.55, 98.59, 98.77])
clean_test4 = np.asarray([ 9.41, 96.80, 97.36, 97.62, 98.05, 98.17, 98.22, 98.44, 98.32, 98.60])
clean_test5 = np.asarray([ 8.19, 96.32, 97.42, 97.74, 97.87, 98.19, 98.20, 98.31, 98.42, 98.57])
clean_test  = np.vstack((clean_test1, clean_test2, clean_test3, clean_test4, clean_test5))

x = np.asarray([0,1,2,3,4,5,6,7,8,9])
fig, ax1 = plt.subplots()
ax1.set_xlabel('Round')
ax1.set_ylabel('Accuracy')
ax1.plot(x, aggregated_test.mean(0), label = 'Aggregated model, w/ watermark, test acc.')
ax1.fill_between(x, aggregated_test.mean(0) - aggregated_test.std(0), aggregated_test.mean(0) + aggregated_test.std(0), alpha=0.2)

ax1.plot(x, aggregated_wm.mean(0), label = 'Aggregated model, w/ watermark, wm acc.')
ax1.fill_between(x, aggregated_wm.mean(0) - aggregated_wm.std(0), aggregated_wm.mean(0) + aggregated_wm.std(0), alpha=0.2)

ax1.plot(x, clean_test.mean(0), label = 'Aggregated model, wo/ watermark, test acc.', linestyle='dashed')
ax1.fill_between(x, clean_test.mean(0) - clean_test.std(0), clean_test.mean(0) + clean_test.std(0), alpha=0.2)

#line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
ax1.axhline(y=43.0, c='black', lw=1.0, linestyle='dotted')
ax1.legend()
plt.savefig('mnist.pdf' , bbox_inches='tight') 
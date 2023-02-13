#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import matplotlib.style as style
import seaborn as sns
from scipy import stats
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

style.use('seaborn-poster')
sns.set(rc={'figure.figsize':(6,4)})
sns.set(font_scale=1.5)

with open('saved_models/classic_cv/simmatch_imagenet_100k_0/log.txt') as f:
        dlines1 = f.readlines()

with open('imagenet.txt') as f:
        flines1 = f.readlines()

with open('imagenet-flex.txt') as f:
        fllines1 = f.readlines()
        
iters = []
dloss1 = []
dacc1 = []
floss1 = []
facc1 = []
flloss1 = []
flacc1 = []
for line in dlines1:
	splits = line.split(',')
	for split in splits:
		if 'iteration' in split:
			iters.append(int(split.split(' ')[2]))
		if 'eval/loss' in split:
			dloss1.append(float(split.split(':')[1][8:]))
		if 'top-1-acc' in split:
			dacc1.append(float(split.split(':')[1]))

for line in flines1:
	splits = line.split(',')
	for split in splits:
		if 'eval/loss' in split:
			floss1.append(float(split.split(':')[1][8:]))
		if 'top-1-acc' in split:
			facc1.append(float(split.split(':')[1]))

for line in fllines1:
	splits = line.split(',')
	for split in splits:
		if 'eval/loss' in split:
			flloss1.append(float(split.split(':')[1][8:]))
		if 'top-1-acc' in split:
			flacc1.append(float(split.split(':')[1]))

dacc1 = list(map(lambda x: x * 100, dacc1))
facc1 = list(map(lambda x: x * 100, facc1))
flacc1 = list(map(lambda x: x * 100, flacc1))

fig1, ax1 = plt.subplots(1, 1)
ax1.plot(iters, facc1, label='FixMatch', color = 'coral', linewidth=2)
ax1.plot(iters, flacc1, label='FlexMatch', color = 'orange', linewidth=2)
ax1.plot(iters, dacc1, label='SequenceMatch', color = 'cornflowerblue', linewidth=2)
ax1.legend(loc=4)
ax1.grid()
# ax1.set(ylim=(0.0, 1.0))
ax1.xaxis.set_major_formatter(ticker.EngFormatter())
ax1.set_xlabel('Iter.')
ax1.set_ylabel('Accuracy (%)')
# ax1.set_title('')

plt.tight_layout(pad=0.4)
plt.grid()
plt.savefig('acc-imagenet.pdf', format='pdf', dpi=1000, tight_layout=True)


fig2, ax2 = plt.subplots(1, 1)
ax2.plot(iters, floss1, label='FixMatch', color = 'coral', linewidth=2)
ax2.plot(iters, flloss1, label='FlexMatch', color = 'orange', linewidth=2)
ax2.plot(iters, dloss1, label='SequenceMatch', color = 'cornflowerblue', linewidth=2)
ax2.legend()
ax2.grid()
ax2.set(ylim=(2.0, 4.0))
ax2.xaxis.set_major_formatter(ticker.EngFormatter())
ax2.set_xlabel('Iter.')
ax2.set_ylabel('Loss')
# ax2.set_title('CIFAR-10-120')

plt.tight_layout(pad=0.4)
plt.grid()
plt.savefig('loss-imagenet.pdf', format='pdf', dpi=1000, tight_layout=True)

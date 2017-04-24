#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

from model import MANet,VGGNet


if __name__ == '__main__':

	datasetFiles = ['/Users/yanzheng/Downloads/GTAVDataset/dataset.txt']
	
	Net = MANet()

	dataset = Net.toSequenceDataset(datasetFiles)	
	dataset = [sample for sample in dataset]
	
	valLen = int(len(dataset)*0.1)
	valDataset = dataset[0:valLen]
	dataset = np.delete(dataset, np.s_[0:valLen], 0)

	trainGenerator = Net.dataGenerator(dataset)
	valGenerator = Net.dataGenerator(valDataset)

	model = Net.getModel()
	model.compile(optimizer=RMSprop(), loss='mse', clipnorm=1.0)
	ckp_callback = ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True, save_weights_only=True, mode='min')
	
	model.fit_generator(
		trainGenerator,
		samples_per_epoch=len(dataset),
		nb_epoch=1000,
		validation_data=valGenerator,
		nb_val_samples=len(valDataset),
		callbacks=[ckp_callback]
	)

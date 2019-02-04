#!/usr/bin/env python3

from __future__ import division

import random
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy import ndimage
from scipy import stats
from scipy.spatial import ConvexHull
import numba
import multiprocessing
import itertools
import sys
import glob, os
import gzip
import shutil

from collections import Counter

# function to read a ASCII raster map in python
def readMap_nonASCII(filename): 
	memMap = []
	filemap = open(filename, 'r')
	tolist = [line.rstrip() for line in filemap] # remobe '\n' at the end of the line!
	mydatnospace = [line.split(' ') for line in tolist] # ascii file are space separated
	mydatalist = [line[0:len(line)] for line in mydatnospace] # ascii file are space separated
	mydata = np.array(mydatalist) 
	mydata = mydata.astype(float) # transform mode of data to float
	filemap.close()
	return mydata

def readMap(filename): 
	memMap = []
	filemap = open(filename, 'r')
	tolist = [line.rstrip() for line in filemap] # remobe '\n' at the end of the line!
	mydatalist = [line.split(' ') for line in tolist] # ascii file are space separated
	mydata = np.array(mydatalist[6:]) # transform the list to an array. Ignore first 6 lines of the ASCII file
	mydata[mydata[:]=='NA']=0 # encode NA values
	mydata = mydata.astype(float) # transform mode of data to float
	filemap.close()
	return mydata

# function to transform a float matrix into a integer map
def landscape_int(n):
	prob=n-np.floor(n)
	return(np.random.binomial(1, p=prob))

def bin_ndarray(ndarray, init_resolution, resolution, operation='sum'):
	"""
	Bins an ndarray in all axes based on the target resolution, by summing or averaging.
	Number of output dimensions must match number of input dimensions and new axes must divide old ones.
	Example
	-------
	>>> m = np.arange(0,100,1).reshape((10,10))
	>>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
	>>> print(n)
	"""
	new_shape = np.divide(np.multiply(ndarray.shape,init_resolution),resolution).astype(int) # new size/shape of the array
	operation = operation.lower()
	if not operation in ['sum', 'mean','mode']:
		raise ValueError("Operation not supported.")
	if ndarray.ndim != len(new_shape):
		raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,new_shape))
	compression_pairs = [(d, c//d) for d,c in zip(new_shape,ndarray.shape)]
	flattened = [l for p in compression_pairs for l in p]
	# if the new shape is not compatible with the original shape with need to drop some column. By default we will remove a few column on the east and south part of the map
	nrows = flattened[1]*(ndarray.shape[0]//flattened[1])
	ncols = flattened[3]*(ndarray.shape[1]//flattened[3])
	ndarray_reshape = ndarray[:nrows,:ncols]
	ndarray_reshape = ndarray_reshape.reshape(flattened)
	#ndarray_reshape2 = ndarray_reshape
	for j in range(flattened[0]):
		for k in range(flattened[1]):
			for l in range(flattened[3]):
				values = ndarray_reshape[j][k][l]
				if not np.all(values<0) :
					values[values==-99]=0
					ndarray_reshape[j][k][l] = values
	if operation=='mode':
		reshape_dataF=np.zeros(sum(ndarray_reshape.T).shape)
		tutu = ndarray_reshape.T
		for j in range(flattened[2]):
			for i in range(flattened[3]):	
				for smallarray in range(flattened[0]):	
					myvect=[]
					for bigarray in range(flattened[1]):
						myvect.append(tutu[bigarray][smallarray][i,j])
					myvect=np.array(myvect)
					if not np.all(myvect<0):
						myvect[myvect<0]=0	
					count_data = Counter(myvect)
					mode = max(myvect, key=count_data.get)
					reshape_dataF[smallarray][i,j]=mode
					#reshape_dataF[smallarray][i,j]=sum(myvect)
		reshape_dataS=np.zeros((flattened[0],flattened[0]))
		ndarray_reshape=reshape_dataF.T 
		for smallarray in range(flattened[0]):
			for j in range(flattened[2]):
				myvect=[]
				for i in range(flattened[1]):	
					myvect.append(ndarray_reshape[smallarray][i,j])
				myvect=np.array(myvect)
				if not np.all(myvect<0):
					myvect[myvect<0]=0	
				#reshape_dataS[smallarray,j]=sum(myvect)
				count_data = Counter(myvect)
				mode = max(myvect, key=count_data.get)
				reshape_dataS[smallarray,j]=mode
		ndarray_reshape=np.array(reshape_dataS)
		ndarray_reshape[ndarray_reshape<0]=-99
		return ndarray_reshape
	else:
		for i in range(len(new_shape)):
			op = getattr(ndarray_reshape, operation)
			ndarray_reshape = op(-1*(i+1))
			if i == 0:
				for j in range(flattened[0]):
					#print('here is the issue')
					tutu = ndarray_reshape[j].T
					#print('after tutu')
					for k in range(flattened[2]):
						values = tutu[k]
						if not np.all(values<0):
							#print('one')
							#print(values)
							values[values<0]=0
							#print('two')
							tutu[k] = values
							#print('three')
					ndarray_reshape[j]=tutu.T
		ndarray_reshape[ndarray_reshape<0]=-99
		return ndarray_reshape

# plot habitat suitability as a heat map
def plot(my_habitat):
	my_habitat[my_habitat<0]=-10
	fig = plt.figure(figsize=(14, 8))
	plt.imshow(my_habitat, cmap='hot', interpolation='nearest')
	plt.show()

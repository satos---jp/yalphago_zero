#coding: utf-8

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset


import igo_library

from chainer import cuda

import tkinter as tk

import datetime


board_n = 4
data_t = 3 
#3手*2,手番


class IGO(chainer.Chain):
	def __init__(self):
		super(IGO,self).__init__()
		
		with self.init_scope():
			"""
			#16 :: filterの種類数
			#ksize :: カーネルフィルターの縦横の大きさ
			self.head_l1 = L.Convolution2D(None, 16, ksize=(3,3), stride=1)
			self.head_l2 = L.BatchNormalization((16,2,5))
			
			self.ress = []
			for d in range(6):
				cv = L.Convolution2D(None, 16, ksize=(3,3), stride=1)
				ba = L.BatchNormalization((16,2,5))
				self.ress.append((cv,ba))
			"""
			
			wn = board_n * board_n * data_t
			self.head_l1 = L.Linear(wn)
			self.head_l2 = L.Linear(wn)
			self.head_l3 = L.Linear(wn)
			self.head_l4 = L.Linear(wn)
						
			self.l2 = L.Linear(17)
			self.l3 = L.Linear(1)
	
	def __call__(self,x):
		#ha = F.relu(self.head_l2(self.head_l1(x)))
		
		x = F.sigmoid(self.head_l1(x))
		x = F.sigmoid(self.head_l2(x))
		x = F.sigmoid(self.head_l3(x))
		x = F.sigmoid(self.head_l4(x))
		
		#手番
		h1 = self.l2(x)
		
		#勝敗
		h2 = self.l3(x)
		return (F.softmax(h1),F.tanh(h2))


model = IGO()

serializers.load_npz("chainer_test_model_16_7_5_10times___2018_07_13_10_06_17___1618.npz",model)





#serializers.load_npz("chainer_test_model_16_7_5_10times___2018_07_10_07_14_15___2406.npz",model)

#model.cleargrads()
#np.random.seed(123)

#[黒,白,色]

board = [[([0 for _ in range(0,data_t-1)] + [-1]) for _ in range(0,board_n)] for _ in range(0,board_n)]


cb = [
	[ 0, 0, 0, 0],
	[ 0, 1, 0, 0],
	[ 0, 0, 0, 0],
	[ 0, 0, 0, 0]
]

for y in range(0,4):
	for x in range(0,4):
		if cb[y][x]==1:
			board[y][x][-3]=1
		elif cb[y][x]==-1:
			board[y][x][-2]=1
#board[1][1][-2]=1
#board[2][2][-3]=1
#board[1][2][-3]=1
#board[2][1][-3]=1



p,v = model(np.array([board],dtype=np.float32))

#print(p,v)
#exit(0)
p,v = p.data[0],v.data[0][0]



def p2str(p):
	res = ""
	for y in range(0,board_n):
		for x in range(0,board_n):
			res += ("%.5f " % p[y*board_n+x])
		res += "\n"
	res += ("%.5f\n" % p[-1])
	return res


print(v)
print(p2str(p))


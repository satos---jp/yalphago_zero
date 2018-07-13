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
#data_t = 7 
#3手*2,手番

data_t = 3
#1手番のみでいいやろ(コウができたらそのとき考える)

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

def getstamp():
	return datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

class Chainer_Trainer:
	def __init__(sl):
		sl.model = IGO()
		
		serializers.load_npz("chainer_test_model_16_7_5_10times___2018_07_13_18_01_18___1975.npz",sl.model)
		
		sl.optimizer = chainer.optimizers.SGD().setup(sl.model) #確率的勾配降下法
		sl.mem_lr = sl.optimizer.lr
	
	def train_from_kihu(sl,data):
		def lossfun(arg1,arg2):
			h,w = arg1 #ネットワークから得られた
			gh,gw = arg2 #正解

			v1 = (gw-w)*(gw-w)
			#ここ、論文とh,ghが逆だけど、 log(0)とかをとりたくないし問題ないよね...??
			v2 = F.log(h)[np.arange(gh.size),gh] 
			loss = F.sum(v1) - F.sum(v2)
			#loss = F.sum(v1)
			return loss
		
		sl.optimizer.lr = sl.mem_lr
		batch_size = 10
		for epoch in range(300):
			perm = np.random.permutation(len(data))
			ls = 0
			loss_sum = 0; acc_sum = 0
			n_batches = (len(data) + batch_size - 1) // batch_size
			for i in range(n_batches):
				x = np.array([data[i][0] for i in perm[i * batch_size : (i + 1) * batch_size]],dtype=np.float32)
				hand_t = np.array([data[i][1] for i in perm[i * batch_size : (i + 1) * batch_size]],dtype=np.int)
				win_t = np.array([data[i][2] for i in perm[i * batch_size : (i + 1) * batch_size]],dtype=np.float32)
				
				#print(x.shape,t.shape)
				model.cleargrads()
				y = model(x)
				loss = lossfun(y,(hand_t,win_t))
				loss.backward()
				#acc = F.accuracy(y, (hand_t,win_t))
				sl.optimizer.update()
				loss_sum += loss.array
				#acc_sum += acc.array
			
			lp = loss_sum / n_batches
			if epoch == 0:
				print('first loss %3.6f' % (lp))
			elif epoch % 5 == 0:
				sl.optimizer.lr *= 0.95
		
		print('loss %3.6f' % (lp))
		serializers.save_npz("chainer_test_model_16_7_5_10times___" + getstamp() + "___" + ("%d" % (lp*100)) + ".npz", sl.model)


np.random.seed(456)

chainer_trainer = Chainer_Trainer()
model = chainer_trainer.model



def isvalid_hand(board,t):
	if t==board_n*board_n:
		return True
	return (board[t//board_n][t%board_n][-2]==0 and board[t//board_n][t%board_n][-3]==0)

def put_for_data(board,t):
	bc = int(np.sign(board[1][1][-1]))
	#print(board,t,c)
	def f(y,x):
		if 0 < y and y <= board_n and 0 < x and x <= board_n:
			if board[y-1][x-1][-3]==1:
				return 1
			elif board[y-1][x-1][-2]==1:
				return -1
			else:
				return 0
		else:
			return 2 #壁
	
	nb = [[f(y,x) for x in range(0,board_n+2)] for y in range(0,board_n+2)]
	#print(nb)
	
	if t!=board_n*board_n:
		igo_library.put(nb,(t//board_n)+1,(t%board_n)+1,bc)
		#print('to_nb',nb)
	
	def g(y,x): #最後にcが打った状態、なので、次の手番は-c
		#print(y,x,nb[y+1][x+1],(1 if nb[y+1][x+1]==1 else 0))
		return (
			board[y][x][2:-1] + [
				(1 if nb[y+1][x+1]==1 else 0),
				(1 if nb[y+1][x+1]==-1 else 0),
				-bc
			]
		)
	
	res = [[g(y,x) for x in range(0,board_n)] for y in range(0,board_n)]
	#print('board',board)
	#print('res',res)
	return res


def learnboard2str(b):
	tb = map(lambda s: map(lambda v: 
		'黒' if v[-3]==1 else
		'白' if v[-2]==1 else
		'＋'
	,s),b)
	return '\n'.join(map(lambda s: ''.join(s),tb))



class MC_Tree:
	def __init__(sl,board):
		sl.board = board
		p,v = model(np.array([board],dtype=np.float32)) # 手確率,盤面価値
		sl.P,sl.v = p.data[0],v.data[0][0]
		
		#ここでPに乱数を加える(そうしないと新たな手を打たない)
		
		sl.P = sl.P + (np.random.random(board_n*board_n+1) * 0.03)
		
		#print(sl.P)
		
		sl.children = [None for _  in range(0,board_n*board_n+1)]
		sl.N = 1 #ノードを訪れた回数
		sl.W = sl.v #評価値のSum
		sl.Q = sl.W / sl.N
		
		sl.c = int(np.sign(board[0][0][-1]))
		
	#次の手を返す
	def select(sl):
		# argmax Q+Uを選ぶ。ここで、
		# U = c*P*sqrt(ΣNb)/(1+Na)
		# cは定数
		
		C = 0.1 #適当
		
		maxv = -1
		rd = -1
		Sqn = np.sqrt(sl.N)
		for d,nd in enumerate(sl.children):
			if not isvalid_hand(sl.board,d):
				continue
			if nd==None:
				qu = 0 + C * sl.P[d] * Sqn
			else:
				qu = sl.c * nd.Q + C * sl.P[d] * Sqn / (1.0 + nd.N) #手番に評価をかけるべき(手番1なら評価1を、手番-1なら評価-1を選ぶべき)
			
			#print(d,qu,maxv)
			if qu > maxv:
				maxv = qu
				rd = d
		
		return rd
	
	def exploler(sl):
		d = sl.select()
		if sl.children[d] == None:
			#leafなので展開する	
			to_node = MC_Tree(put_for_data(sl.board,d))
			sl.children[d] = to_node
			v = to_node.v
		else:
			#バックアップ
			v = sl.children[d].exploler()
		
		sl.N += 1
		sl.W += v
		sl.Q = sl.W / sl.N
		return v
	
	def nexthand(sl):
		#とりあえず最大Nの手を返す(この割合は学習状況によって変えるらしい)
		#(論文では最初の30手はNの比率で確率的に返す
		# それ以降は最大のNを返す)
		
		rn = 0
		rd = -1
		for d,nd in enumerate(sl.children):
			if not isvalid_hand(sl.board,d):
				continue
			if nd!=None and nd.N > rn:
				rn = nd.N
				rd = d
		return rd
	
	def dump_str(sl,tn=0):
		def p(a,s):
			print(('\n' + s).replace('\n','\n' + a * '  ')[1:])
		p(tn,"%d : %d %f" % (sl.c,sl.N,sl.W))
		p(tn,learnboard2str(sl.board))
		for d,nd in enumerate(sl.children):
			if nd!=None:
				p(tn+1,"%d : %d " % ((d//board_n)+1,(d%board_n)+1))
				nd.dump_str(tn+2)

def p2str(p):
	res = ""
	for y in range(0,board_n):
		for x in range(0,board_n):
			res += ("%.5f " % p[y*board_n+x])
		res += "\n"
	res += ("%.5f\n" % p[-1])
	return res

def pubc(b):
	root = MC_Tree(b)
	for t in range(0,500):
		#print(t)
		#print('dump root')
		#root.dump_str()
		root.exploler()
	
	#print('dump root')
	#root.dump_str()
	#print(learnboard2str(root.board))
	print(root.v)
	print(p2str(root.P))
	return root.nexthand()



if __name__=='__main__':
	import gui
	
	gc = 1
	def func(board,y,x):
		global gc
		def f(y,x):
			if 0 < y and y <= board_n and 0 < x and x <= board_n:
				return board[y-1][x-1]
			else:
				return 2 #壁
		
		nb = [[f(y,x) for x in range(0,board_n+2)] for y in range(0,board_n+2)]
		igo_library.put(nb,y+1,x+1,gc)
		gc *= -1
		
		
		tb = [[([0 for _ in range(0,data_t-1)] + [gc]) for _ in range(0,board_n)] for _ in range(0,board_n)]
		#print(tb)
		for y in range(0,4):
			for x in range(0,4):
				if nb[y+1][x+1]==1:
					tb[y][x][-3]=1
				elif nb[y+1][x+1]==-1:
					tb[y][x][-2]=1
		
		print(tb)
		ha = pubc(tb)
		ty,tx = (ha//board_n),(ha%board_n)
		print(ty,tx)
		if ha!=board_n*board_n:
			igo_library.put(nb,ty+1,tx+1,gc)
		gc *= -1
		res = [[nb[y+1][x+1] for x in range(0,board_n)] for y in range(0,board_n)]
		
		return res
	
	cb = [
		[ 0, 0, 0, 0],
		[ 0, 0, 0, 0],
		[ 0, 0, 0, 0],
		[ 0, 0, 0, 0]
	]
	
	"""
	cb = [
		[ 1, 0, 1, 0],
		[ 0, 1,-1,-1],
		[ 1,-1, 1, 0],
		[ 0,-1, 1, 0]
	]
	"""

	#pubc(board)
	
	gui.Tk_board(cb).run(func)
	
	
	exit(0)


def winner(board):
	def f(y,x):
		if board[y][x][-3]==1:
			return 1
		elif board[y][x][-2]==1:
			return -1
		else:
			return 0
	b = [[f(y,x) for x in range(0,board_n)] for y in range(0,board_n)]
	
	
	def touch_check(b,gone,y,x):
		gone[y][x] = 1
		tb,tw = False,False
		for dy,dx in [(1,0),(0,1),(-1,0),(0,-1)]:
			ty,tx = y+dy,x+dx
			if tx<0 or ty<0 or board_n<=tx or board_n<=ty or gone[ty][tx]==1:
				continue
			if b[ty][tx]==1:
				tb = True
			elif b[ty][tx]==-1:
				tw = True
			else:
				gb,gw = touch_check(b,gone,ty,tx)
				tb |= gb
				tw |= gw
		return tb,tw
	
	def cont_num(b,gone,y,x):
		gone[y][x] = 2
		res = 1
		for dy,dx in [(1,0),(0,1),(-1,0),(0,-1)]:
			ty,tx = y+dy,x+dx
			if tx<0 or ty<0 or board_n<=tx or board_n<=ty or gone[ty][tx]==2:
				continue
			if b[ty][tx]==0:
				res += cont_num(b,gone,ty,tx)
		return res
	
	gone = [[0 for x in range(0,board_n)] for y in range(0,board_n)]
	bn,wn = 0,0
	for y in range(0,board_n):
		for x in range(0,board_n):
			if b[y][x]==1:
				bn += 1
			elif b[y][x]==-1:
				wn += 1
			else:
				if gone[y][x]!=0:
					continue
				tb,tw = touch_check(b,gone,y,x)
				if tb and tw:
					continue
				cn = cont_num(b,gone,y,x)
				if tb:
					bn += cn
				else:
					wn += cn
		
	if bn > wn:
		return 1
	elif bn < wn:
		return -1
	else:
		return 0


def generate_kihudata():
	kihudata = []
	for tt in range(0,100):
		board_data = [[([0 for _ in range(0,data_t-1)] + [1]) for _ in range(0,board_n)] for _ in range(0,board_n)]
		
		boards = []
		
		lastpass = False
		for t in range(0,30):
			ha = pubc(board_data)
			#boards.append((board_data,[1 if ha==i else 0 for i in range(0,board_n*board_n+1)])) #この盤面ではこれを打ったという学習結果
			#タグはインデクスのみにする
			boards.append((board_data,ha))
			
			board_data = put_for_data(board_data,ha)
			if ha==board_n*board_n:
				if lastpass:
					break
				lastpass = True
			else:
				lastpass = False
						
			if tt==0:
				print(learnboard2str(board_data))
				print('play %d' % t)
		
		w = winner(board_data)
		if tt==0:
			print('win',w)
		kihudata += [(b,h,[w]) for (b,h) in boards]
		
		#print(getstamp())
		#if tt%1==0:
		#	print('generate %d kihu data' % tt)
	
	print('generated kihudata')
	return kihudata



def alpha_zero_loop():
	#for _ in range(0,1):
	while True:
		kihudata = generate_kihudata()
		chainer_trainer.train_from_kihu(kihudata)


"""
from line_profiler import LineProfiler
profiler = LineProfiler()
profiler.add_module(Chainer_Trainer)
profiler.add_module(MC_Tree)
profiler.add_function(alpha_zero_loop)
profiler.runcall(alpha_zero_loop)

profiler.print_stats()
"""

alpha_zero_loop()

"""
class Tk_board:
	def __init__(sl,board):
		sl.board = board
	
	def run(sl,func):
		root = tk.Tk()

		frame = root

		img = tk.PhotoImage(file='yonroban.png')

		canvas = tk.Canvas(root,width=500,height=500)
		canvas.create_image(250,250,image=img)
		canvas.pack()
		
		sl.cs = []
		
		def callback(ev):
			#70,70 -> 440,440
			#1マス46.25
			w = 46.25
			d = 70
			x = (ev.x - d + (w/2)) // w
			y = (ev.y - d + (w/2)) // w
		
			if (0 <= x and x < board_n and 0 <= y and y < board_n) or True:
				#print('put',x,y)
				x,y = int(x+0.5),int(y+0.5)
				sl.board = func(sl.board,y,x)
				
				for bc in sl.cs:
					canvas.delete(bc)
				
				sl.cs = []
				for y in range(0,board_n):
					for x in range(0,board_n):
						if sl.board[y][x]!=0:
							cx,cy = x*w+d,y*w+d
							a = 20
							tc = canvas.create_oval(cx-a,cy-a,cx+a,cy+a,fill=('white' if sl.board[y][x]==-1 else 'black'))
							#print('put',y,x,sl.board[y][x])
							sl.cs.append(tc)
		
		frame.bind("<Button-1>",callback)

		root.geometry('500x500')
		root.mainloop()


Tk_board([[0 for x in range(0,board_n)] for y in range(0,board_n)]).run(func)


"""

	
	

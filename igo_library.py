#coding: utf-8

#学習時には超コウの問題は無視する

board_n = 4

def board2str(b):
	tb = map(lambda s: map(lambda v: 
		'＃' if v==2 else 
		'黒' if v==1 else
		'白' if v==-1 else
		'＋'
	,s),b)
	return '\n'.join(map(lambda s: ''.join(s),tb))

def dfs_check(b,gone,t,y,x,c):
	#print(y,x,c)
	gone[y][x] = t
	res = True
	for dy,dx in [(1,0),(0,1),(-1,0),(0,-1)]:
		ty,tx = y+dy,x+dx
		#print('tytx',ty,tx,b[ty][tx])
		if b[ty][tx]==0:
			return False
		if gone[ty][tx]==t:
			continue
		if b[ty][tx]==c:
			if gone[ty][tx]<t:
				return False
			res = res and dfs_check(b,gone,t,ty,tx,c)
	return res

#盤外は2にする

def dfs_remove(b,y,x,c):
	b[y][x] = 0
	for dy,dx in [(1,0),(0,1),(-1,0),(0,-1)]:
		ty,tx = y+dy,x+dx
		if b[ty][tx]==c:
			dfs_remove(b,ty,tx,c)

def put(b,y,x,c):
	global t
	#print('put',y,x)
	b[y][x] = c
	
	#print(board2str(b))
	
	gone = [[100 for _ in range(0,board_n+2)] for _ in range(0,board_n+2)]
	t = 1
	for dy,dx in [(1,0),(0,1),(-1,0),(0,-1)]:
		ty,tx = y+dy,x+dx
		if b[ty][tx]==-c:
			if gone[ty][tx]<10:
				continue
			#print('check ',ty,tx)
			if dfs_check(b,gone,t,ty,tx,-c):
				dfs_remove(b,ty,tx,-c)
			#print(gone)
			t += 1
	
	#print('check root ',y,x)
	if dfs_check(b,gone,t,y,x,c):
		dfs_remove(b,y,x,c)

#with open("1234.sgf") as f: #58手


def board_strip(b):
	return [[b[y+1][x+1] for x in range(0,board_n)] for y in range(0,board_n)]






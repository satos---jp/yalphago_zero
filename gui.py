#coding: utf-8
import igo_library
import tkinter as tk

board_n = igo_library.board_n


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
			#print(ev.y,ev.x)
			#70,70 -> 440,440
			#1マス46.25
			w = 123.33
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
							a = w/2-10
							tc = canvas.create_oval(cx-a,cy-a,cx+a,cy+a,fill=('white' if sl.board[y][x]==-1 else 'black'))
							#print('put',y,x,sl.board[y][x])
							sl.cs.append(tc)
		
		frame.bind("<Button-1>",callback)

		root.geometry('500x500')
		root.mainloop()


if __name__=="__main__":
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
	
		res = [[nb[y+1][x+1] for x in range(0,board_n)] for y in range(0,board_n)]

		return res


	Tk_board([[0 for x in range(0,board_n)] for y in range(0,board_n)]).run(func)


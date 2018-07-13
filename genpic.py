import matplotlib
from matplotlib import pyplot as plt


plt.figure(figsize=matplotlib.figure.figaspect(1))

for i in range(0,4):
	plt.plot([i,i],[0,3],color='black')
	plt.plot([0,3],[i,i],color='black')

plt.tick_params(labelbottom=False,labelleft=False,color='white')
plt.xlim(0,3)
plt.ylim(0,3)
plt.show()


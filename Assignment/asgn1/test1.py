import time
import matplotlib.pyplot as plt
import numpy as np

i = 0

data = [55, 66] 

fig, ax = plt.subplots()
index = np.arange(1, 3)
bar_width = 0.2

# show the figure, but do not block
plt.show(block=False)

pm, pc = plt.bar(index, data, width=0.2)
pm.set_facecolor('b')
pc.set_facecolor('r')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['Python', 'Numpy'])
ax.set_ylim([0, max(data)*1.2])
ax.set_xlabel('Block '+str(i))
ax.set_ylabel('Time(sec)')
ax.set_title('Block '+str(i)+' Time Comparison')

# plt.savefig('figure'+str(i))
plt.show()
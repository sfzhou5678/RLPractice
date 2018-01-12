# encoding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import math

x = np.arange(1, 100, step=0.1)
# s = [200 / ss for ss in s]
s=3.7/x ** (0.40 * math.sqrt(np.e))
# y = 5 * x
# plt.plot(x, y, '--')
plt.plot(x, s, )
plt.xlabel('Number of labors')
plt.ylabel(r'Elapsed Time')

ax = plt.gca()
# 去掉边框
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# 移位置 设为原点相交
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

target_y = 0.83
target_x = 10

plt.plot([target_x, target_x, ], [0, target_y], 'k--', linewidth=1.5)
plt.plot([0, target_x, ], [target_y, target_y], 'k--', linewidth=1.5)

plt.scatter([target_x, ], [target_y, ], s=40, color='r')
plt.annotate(r'(%d,%.2f)' % (target_x, target_y), xy=(target_x, target_y), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
plt.show()

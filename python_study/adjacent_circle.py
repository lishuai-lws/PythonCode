from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcol
import matplotlib.cm as cm
import matplotlib.patches as mpathes
import math
from matplotlib import colors
from numpy.ma.core import size

labels = range(62)
#生成颜色渐变
cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",['r','b'])
cnorm = mcol.Normalize(vmin=1,vmax=len(labels))
cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
cpick.set_array([])
pie_color = cpick.to_rgba(range(len(labels)),alpha=0.5)
# print(pie_color)
#每一块的数据设置为1
data = [1]*len(labels)
# print(data)
#画图
fig,ax= plt.subplots()
#计算块中心点坐标
angles_labels = [(i+0.5)*2*math.pi/len(labels) for i in range(len(labels))]
coords_labels = [[0.91*math.cos(i),0.91*math.sin(i)]for i in angles_labels]

ax.pie(data,radius=1,labels=labels,colors=pie_color,wedgeprops=dict(width=0.1,edgecolor='black'))
#画线及设置颜色
reds = [0,6.5,3.3,2.2,1.6,1.2,1,0.8,0.65,0.55,0.45,0.35]
for i in range(int(len(labels)/2)+1-len(reds)):
    reds.append(0.1)
# for i in range(len(labels)-len(reds)):
#     reds.append(0.1)
print('reds:',len(reds))
lines_data = [round(random.uniform(0.5,1),2) for i in range(len(reds))]
print(lines_data)
# cm2 = mcol.LinearSegmentedColormap.from_list("LineCmap",['b','r'])
cm2 = plt.cm.Reds
cnorm2 = mcol.Normalize(vmin=0.4,vmax=1)
cpick2 = cm.ScalarMappable(norm=cnorm2,cmap=cm2)
cpick2.set_array([])
for i in range(1,len(reds)):
    ax.annotate("",
            xy=(coords_labels[0][0], coords_labels[0][1]), xycoords='data',
            xytext=(coords_labels[i][0], coords_labels[i][1]), textcoords='data',size=10,    
            arrowprops=dict(arrowstyle="<->",
                            color=cpick2.to_rgba(lines_data[i]),
                            patchB=None,
                            shrinkB=0,
                            connectionstyle=f"arc3,rad={reds[i]}",
                            ),
            )
#显示colorbar
fig.colorbar(cpick2)
# circle = mpathes.Circle([0,0],0.6)
# ax.add_patch(circle)

plt.show()
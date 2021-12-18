import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import math
import numpy as np

def pie_line_plot(datas,labels):
    #生成颜色渐变
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",['r','b'])
    cnorm = mcol.Normalize(vmin=1,vmax=len(labels))
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick.set_array([])
    pie_color = cpick.to_rgba(range(len(labels)),alpha=0.5)
    #每一块的数据设置为1
    pie_data = [1]*len(labels)
    #画图
    fig,ax= plt.subplots()
    #计算块中心点坐标
    angles_labels = [(i+0.5)*2*math.pi/len(labels) for i in range(len(labels))]
    coords_labels = [[0.91*math.cos(i),0.91*math.sin(i)]for i in angles_labels]
    #绘制圆环图
    ax.pie(pie_data,radius=1,labels=labels,labeldistance=1.05,rotatelabels=True,colors=pie_color,
            textprops={'fontsize':5,'fontweight':'light','verticalalignment':'center','horizontalalignment':'center'},wedgeprops=dict(width=0.1,edgecolor='black'))
    #画线及设置颜色,目前只设置了12个距离，其他的距离
    reds = [0,6.5,3.3,2.2,1.6,1.2,1,0.8,0.65,0.55,0.45,0.35]
    for i in range(int(len(labels)/2)+1-len(reds)):
        reds.append(0.1)
    # print('len(reds):',len(reds))
    lines_data = []
    #前面是低位，第二位是高位
    for i in range(len(datas)):
        for j in range(len(datas[0])):
            if datas[i][j] >= 0.5 :
                if i > j:
                    lines_data.append([j,i,datas[i][j]])
                else:
                    lines_data.append([i,j,datas[i][j]])
    # print('lines_data:',len(lines_data))
    #绘制线的根据数值的不同颜色
    cm2 = plt.cm.Reds
    cnorm2 = mcol.Normalize(vmin=0.4,vmax=1)
    cpick2 = cm.ScalarMappable(norm=cnorm2,cmap=cm2)
    cpick2.set_array([])
    # 绘制有关导联间的线
    for line_data in lines_data:
        start,end,data= line_data[1],line_data[0],line_data[2]
        if start-end >len(labels)/2:
            red = reds[len(labels)-start+end]
            start,end = end, start
        else:
            # print(start-end)
            red = reds[start-end]
        ax.annotate("",
                xy=(coords_labels[end][0], coords_labels[end][1]), xycoords='data',
                xytext=(coords_labels[start][0], coords_labels[start][1]), textcoords='data',
                size=10,va="center", ha="center",
                arrowprops=dict(arrowstyle="<->",
                                color=cpick2.to_rgba(data),
                                patchB=None,
                                shrinkB=0,
                                connectionstyle=f"arc3,rad={red}",
                                ),
                )
    #显示colorbar
    fig.colorbar(cpick2)
    #显示图片
    # plt.show()
    plt.savefig('pie_line_v2.pdf')

if __name__ == "__main__":
    # 读取文件
    datas = np.load('res.npy')#邻接矩阵
    #标签
    labels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 
            'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7',
            'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
            'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 
            'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    # datas = np.random.rand(len(labels),len(labels))#测试
    # for i in range(len(datas)):
    #     for j in range(len(datas[0])):
    #         n = np.random.rand()
    #         if n>0.01:
    #             # print(i,j)
    #             datas[i][j]=0
    pie_line_plot(datas,labels)


# coding: utf-8

# # Source location privacy in Cyber-physical systems
# 
# - 节点
# - 链路
# - 网络

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import Queue
import time


# # 1 虚假源调度算法

# ## 公共通用参数与函数

# In[3]:

# # 0 公共通用参数与模型

class commonPM:
    """
    1) isConnected(Network): 网络连通性判定
    2) calculate2Distance(node1, node2): 节点间距离计算
    3) delayModel(Network, u, v): 单跳时延模型
    4) consumeEnergyModel(Network, node): 节点能耗模型
    5) pairEnergyModel(U, V): U -> V 能耗模型
    """
    def __init__(self):
         self.use = "公共通用参数与模型"

    def isConnected(self, Network, src):
        "判定一个网络是否连通"
        print "网络连通性判定..."
        numConnected = 0
        
        queue = Queue.Queue()
        visited = np.zeros(Network.nodeNumber, dtype=np.int8)
        queue.put(src)  # 从 sink 开始广搜整个网络
        visited[src] = 1
        numConnected += 1
        while not queue.empty():
            u = queue.get()
            U = Network.nodeList[u]
            for V in Network.nodeList:
                v = V.identity
                if u != v and self.calculate2Distance(U, V) < U.radius:
                    U.adj.add(v)
                    if visited[v] == 0:
                        queue.put(v)
                        visited[v] = 1
                        numConnected += 1
        if numConnected == Network.nodeNumber:
            return True
        return False

    def calculate2Distance(self, node1, node2):
        "计算两个节点之间的欧式距离"
        x1 = node1.position[0]
        y1 = node1.position[1]
        x2 = node2.position[0]
        y2 = node2.position[1]
        return np.sqrt((x1-x2)**2+(y1-y2)**2)

    
    def delayModel(self, Network, u, v):
        """
        估计单个节点，单跳，单位 bit 数据的时延
        """
        tuv = 100.       # 无干扰,bit,接收时间 ns
        pe = 0           # 环境的噪声功率
        guvl = 9.488e-5  # 无线设备参数，d < d0
        guvh = 5.0625    # 无线设备参数，d >= d0
        d0 = 231         # 跨空间距离阈值
        pt = 10e0        # 节点发送功率
        U = Network.nodeList[u]
        V = Network.nodeList[v]
        pv = 0           # 接收节点干扰功率
        for neighbor in V.adj:
            Neighbor = Network.nodeList[neighbor]
            if neighbor == u or (Neighbor.state == "NORMAL" and Neighbor.weight == 0):
                continue
            else:
                d = self.calculate2Distance(Neighbor, V)
                if d < d0:
                    puv = (guvl*pt)/(d**2)
                else:
                    puv = (guvh*pt)/(d**4)
                pv += puv
        puv = 0          # u->v 接收功率
        d = self.calculate2Distance(U, V)
        if d < d0:
            puv = (guvl*pt)/(d**2)
        else:
            puv = (guvh*pt)/(d**4)
        ruv = 0          # 信干噪比
        if pv > 1e-32:
            ruv = puv/(pe+pv)
        else:
            ruv = 20.
        # 范围单跳，单位 bit 时延
        return tuv/(1.-np.exp(ruv*(-0.5)))
    
    def pairDelayModel(self, U, V):
        """
        单通道链路传输时延
        """
        a = 1 
        
    def energyModel(self, G, node):
        """
        估计单个节点，单位 bit 数据，的能耗
        """
        d0 = 231         # 跨空间距离阈值
        d = node.radius  # 节点广播半径
        Eelec = 50.      # 单位 bit 数据能耗
        c_fs = 10e-3     # 自由空间下单位放大器能耗
        c_mp = 0.0013e-3 # 存在多径损耗打单位放大器能耗
        # 接收能耗
        rE = 0           # 单位数据接收能耗
        flag = False     # broadcast 标志位
        for neighbor in node.adj:
            # neighbor send message to node.identity
            if G.adjacentMatrix[neighbor, node.identity] == 1:
                rE += Eelec
            # 广播消息标志
            if G.adjacentMatrix[node.identity, neighbor] == 1:
                flag = True
        # 发送能耗
        tE = 0           # 单位数据发送能耗
        if flag:
            tE += Eelec
            if d < d0:
                tE += c_fs*(d**2)
            else:
                tE += c_mp*(d**4)
        return rE + tE
    def pairEnergyModel(self, U, V):
        """
        u -> v 能耗
        """
        d0 = 231         # 跨空间距离阈值
        d = U.radius     # 节点广播半径
        Eelec = 50.      # 单位 bit 数据能耗
        c_fs = 10e-3     # 自由空间下单位放大器能耗
        C_mp = 0.0013e-3 # 存在多径损耗打单位放大器能耗
        # u send consumption
        tE = 0           # 单位数据发送能耗
        tE += Eelec
        if d < d0:
            tE += c_fs*(d**2)
        else:
            tE += c_mp*(d**4)
        
        # v receive consumption
        rE = 0           # 单位数据接收能耗
        rE += Eelec
        return tE, rE


# ## 1.1 网络模型
# 
# - 节点
# - 链路
# - 网络

# In[ ]:

class cpsNode:
    """
    Cyber-physical systems 节点：
    1）身份    2）位置    
    3）能量    4）通信半径  
    5）状态    6）权重
    7）level
    8）邻居列表
    9）被捕获似然
    
    state: SINK -> 基站
           SOURCE -> 源节点
           NORMAL -> 普通节点
           BACKBONE -> 骨干节点
           FAKE -> 虚假源
    
    --------------------
    变量成员：
    identity = 身份/编号
    position = 位置
    energy   = 能量
    radius   = 通信半径
    state    = 状态信息
    weight   = 权重信息，表示节点是否广播消息
    level    = 层次信息，用于描述与 sink 间打跳数
    adj      = 相邻节点集合
    --------------------
    方法成员：
    dispaly()
    """
    def __init__(self, identity=-1, position=(0,0), energy=1e8, radius=20, state=-1, weight=-1):
        self.identity = identity
        self.position = position
        self.energy = energy
        self.radius = radius
        self.state = state
        self.weight = weight   # 1:broadcast, 0:not broadcast
        self.level = -1 if state != "SINK" else 0
        self.adj = set()
    def display(self):
        print "Node", self.identity, ":", self.position, self.energy, self.radius, self.state, self.weight

class cpsLink:
    """
    Cyber-physical systems 链路
        1）nodeFrom    2）nodeTo
        3）权重
    """
    def __init__(self, nodeFrom=cpsNode(), nodeTo=cpsNode(), weight=-1):
        self.nodeFrom = nodeFrom
        self.nodeTo = nodeTo
        self.weight = weight
    def display(self):
        print "Link:\n", self.nodeFrom.identity, self.nodeTo.identity, self.weight
    
class cpsNetwork:
    """
    Cyber-physical systems 网络
    1）nodeNumber    2）areaLength
    3）initEnergy    4）nodeList      
    5）0/1 adjacentMatrix
    --------------------
    变量成员：
    nodeNumber = 节点数量
    areaLength = 区域边长
    initEnergy = 节点初始能量(?)
    radius     = 节点通信半径(?)
    nodeList   = 节点列表
    --------------------
    方法成员：
    __init__(nodeNumber, areaLength, initEnergy, radius)
    display()
    """
    def __init__(self, nodeNumber=500, areaLength=1000, initEnergy=1e6, radius=50):
        self.nodeNumber = nodeNumber
        self.areaLength = areaLength
        self.initEnergy = initEnergy
        self.radius = radius
        self.nodeList = []   # [0, nodeNuber-1] are NORMAL nodes
        for i in range(nodeNumber):
            position = (np.random.rand(2) - 0.5) * areaLength # [-0.5*areaLength, 0.5*areaLength)
            position = (position[0], position[1])
            self.nodeList.append(cpsNode(identity=i, position=position, energy=initEnergy, radius=radius, state="NORMAL"))
            
        self.adjacentMatrix = np.zeros((nodeNumber, nodeNumber), dtype=np.int8)
    def display(self):
        print "节点总数：", self.nodeNumber
        print "正方形区域边长：", self.areaLength
        print "节点初始能量：", self.initEnergy
        temp_x = []
        temp_y = []
        for i in range(self.nodeNumber):
            self.nodeList[i].display()
            temp_x.append(self.nodeList[i].position[0])
            temp_y.append(self.nodeList[i].position[1])
        print self.adjacentMatrix
        plt.figure(figsize=(4,4))
        plt.plot(temp_x, temp_y, 'yp')
        plt.axis("equal")
        plt.show()


# ## 1.2 攻击者模型

# In[ ]:

class cpsAttacker:
    """
    攻击者模型
    ---------
    变量成员：
    position = 攻击者位置，cpsNode 类型
    trace    = 攻击者的历史轨迹信息，位置用节点编号表示
    ---------
    方法成员：
    __init__(position)     = 初始化
    display()              =
    traceBack(backbone, G) = weight == 1 or backbone 作为可回溯的位置
    tracebackNetwork(G)    = weight == 1 作为可回溯的位置
    """
    def __init__(self, position=cpsNode()):
        self.position = position
        self.trace = []
    def display(self):
        self.position.display()
        print self.trace
        
    def traceBack(self, backbone, G):
        """攻击者移动模式1 - 随机选择位置 weight==1 or backbone"""
        bestposition = -1
        bestNode = -1
        for node in self.position.adj:
            Node = G.nodeList[node]
            if Node.weight == 1 or (node in backbone):
                RAND = np.random.rand()
                if RAND > bestposition:
                    bestposition = RAND
                    bestNode = node
                    self.position = Node
                else:
                    pass
        if bestNode != -1:
            self.trace.append(bestNode)
        else:
            self.trace.append(self.position.identity)
    def tracebackNetwork(self, G):
        """攻击者移动模式2 - 随机选择位置 weight==1"""
        bestProbability = -1
        bestNode = -1
        u = self.position.identity
        for node in self.position.adj:
            Node = G.nodeList[node]
            if G.adjacentMatrix[u, node] == 1 or G.adjacentMatrix[node, u] == 1:
                RAND = np.random.rand()
                if RAND > bestProbability:
                    bestProbability = RAND
                    bestNode = node
                else:
                    pass
        if bestNode != -1:
            self.position = G.nodeList[bestNode]
            self.trace.append(bestNode)
        else:
            self.trace.append(self.position.identity)
            
    def tracebackUsingAdjMatrix(self, G):
        """
        攻击者移动模式 3:
        随机选择位置 G.adjacentMaxtrix[u,v] == 1
        """
        bestposition = -1
        bestNode = -1
        u = self.position.identity
        for v in self.position.adj:
            V = G.nodeList[v]
            if G.adjacentMatrix[u, v] == 1:
                RAND = np.random.rand()
                if RAND > bestposition:
                    bestposition = RAND
                    bestNode = v
                    self.position = V
                else:
                    pass
        if bestNode != -1:
            self.trace.append(bestNode)
        else:
            self.trace.append(self.position.identity)      


# # 动画制作与保存

# In[ ]

"""
from JSAnimation import IPython_display
from matplotlib import animation
from matplotlib import patchesid you know that you can close tabs in the editor and the tool windows of PyCharm Community Edition without actually using the context menu commands? It is enough to point with your mouse cursor to a tab to be closed, and click the middle mouse button, or just use the Shift+click combination.
"""


# In[ ]:id you know that you can close tabs in the editor and the tool windows of PyCharm Community Edition without actually using the context menu commands? It is enough to point wid you know that you can close tabs in the editor and the tool windows of PyCharm Community Edition without actually using the context menu commands? It is enough to point with your mouse cursor to a tab to be closed, and click the middle mouse button, or just use the Shift+click combination.ith your mouse cursor to a tab to be closed, and click the middle mouse button, or just use the Shift+click combination.

"""
fig = plt.figure(figsize=(12,5))
plt.axis('equal')
global e
x = fs.G.nodeList[fs.attacker.trace[0]].position[0]
y = fs.G.nodeList[fs.attacker.trace[0]].position[1]
e = patches.Ellipse(xy=[x, y], width=fs.radius, height=fs.radius, facecolor='w', edgecolor='r')

ax1 = plt.subplot(1,2,1)
ax1.set_xlim(-fs.areaLength/2., fs.areaLength/2.)
ax1.set_ylim(-fs.areaLength/2., fs.areaLength/2.)
line1, = plt.plot([], [], 'yo')   # 节点
line11, = ax1.plot([], [], 'gs')  # 骨干节点
line12, = ax1.plot([], [], 'g')   # 骨干网络关系
line13, = ax1.plot([], [], 'bs')  # 虚假源
line14, = ax1.plot([], [], 'rp')  # 攻击者
ax1.add_artist(e)

ax2 = plt.subplot(2,2,2)
ax2.set_xlim(1, fs.safety+10)
ax2.set_ylim(min(fs.listDelay)/2., max(fs.listDelay)*2.)
line2, = ax2.semilogy([], [], 'k')    # 时延曲线
plt.legend(['delay'])

ax3 = plt.subplot(2,2,4)
ax3.set_xlim(1, fs.safety+10)
ax3.set_ylim(min(fs.listEnergyConsumption)/2., max(fs.listEnergyConsumption)*2.)
line3, = ax3.semilogy([], [], 'k')    # 能耗曲线
plt.legend(['Energy Consumption'])

def init():
    line1.set_data([], [])
    line11.set_data([], [])
    line12.set_data([], [])
    line13.set_data([], [])
    line14.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line11, line12, line13, line14, line2, line3,
def animate(i):
    # 节点
    x = [node.position[0] for node in fs.G.nodeList if node.identity not in fs.backbone]
    y = [node.position[1] for node in fs.G.nodeList if node.identity not in fs.backbone]
    line1.set_data(x, y)
    # 骨干节点
    x = [fs.G.nodeList[v].position[0] for v in fs.backbone]
    y = [fs.G.nodeList[v].position[1] for v in fs.backbone]
    line11.set_data(x, y)
    # 骨干网络关系
    line12.set_data(x, y)
    # 虚假源
    x = [fs.G.nodeList[v].position[0] for v in fs.listFakeSource[i]]
    y = [fs.G.nodeList[v].position[1] for v in fs.listFakeSource[i]]
    line13.set_data(x, y)
    # 攻击者位置
    global e
    ax1.artists.remove(e)
    x = fs.G.nodeList[fs.attacker.trace[i]].position[0]
    y = fs.G.nodeList[fs.attacker.trace[i]].position[1]
    line14.set_data([x], [y])
    e = patches.Ellipse(xy=[x, y], width=fs.radius, height=fs.radius, facecolor='w', edgecolor='r')
    ax1.add_artist(e)
    # 时延曲线
    x = list(range(1, i+2))
    y = fs.listDelay[:i+1]
    line2.set_data(x, y)
    # 能耗曲线
    x = list(range(1, i+2))
    y = fs.listEnergyConsumption[:i+1]
    line3.set_data(x, y)
    return line1, line11, line12, line13, line14, line2, line3,
animation.FuncAnimation(fig, animate, init_func=init, frames=20, interval=1000, blit=True)
# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=fs.safety, interval=1000, blit=True)
# Writer = animation.writers["ffmpeg"]
# writer = Writer(fps=1, metadata=dict(artist="wangrui"), bitrate=1800)
# anim.save("FakeSourceScheduling.mp4", writer=writer)
# plt.show()
"""


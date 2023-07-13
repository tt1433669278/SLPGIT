
# coding: utf-8

# # 对比算法
# * phantom routing
# * 基于树结构的诱导路由技术
# * 动态虚假源选择算法

# In[1]:

from BaseSLPinCPS import *


# ## Phantom routing

# In[147]:

class routingPhantom(commonPM):
    def __init__(self, G = cpsNetwork(nodeNumber=10, areaLength=20, initEnergy=1e6, radius=50),                  Tmax=1000, Hwalk=30, sink_pos=(0,0), source_pos=(0.45*20, 0.4*20)):
        self.G = G
        self.Tmax = Tmax
        self.Hwalk = Hwalk
        self.sink = -1
        self.sink_pos = (0,0)
        self.source = -1
        self.source_pos = (G.areaLength*0.45, G.areaLength*0.45)
        self.attacker = cpsAttacker()
        self.safety = -1
        self.listDelay = []
        self.listEnergyConsumption = []
        self.listPath = []
    def display(self):
        print "节点总数 ", self.G.nodeNumber
        print "区域边长 ", self.G.areaLength
        print "节点初始能量 ", self.G.initEnergy
        print "节点最大通信半径 ", self.G.radius
        print "最大周期数 ", self.Tmax
        print "随机步数 ", self.Hwalk
        print "sink ", self.sink, self.sink_pos
        print "source ", self.source, self.source_pos
        
    def generateSINKandSOURCE(self):
        """
        增加 sink 和 source，并更新邻接矩阵元素个数(+2)
        其中，
        sink 位于区域中心，(0,0)
        source 位于区域右上角 (0.45*areaLength, 0.45*areaLength)
        """
        self.sink = self.G.nodeNumber
        self.source = self.G.nodeNumber+1
        self.G.nodeList.append(cpsNode(identity=self.G.nodeNumber,                                        position=self.sink_pos,                                        energy=self.G.initEnergy*100., radius=self.G.radius,                                        state="SINK"))
        self.G.nodeList.append(cpsNode(identity=self.G.nodeNumber+1,                                        position=self.source_pos,                                        energy=self.G.initEnergy, radius=self.G.radius,                                        state="SOURCE"))
        self.G.adjacentMatrix = np.zeros((self.G.nodeNumber+2, self.G.nodeNumber+2), dtype=np.int8)
        self.G.nodeNumber += 2
    
        
    def deployAttacker(self, node):
        """
        部署攻击者初始位置，位于 sink
        """
        while self.attacker.trace:
            self.attacker.trace.pop()
        self.attacker.position = node
        self.attacker.trace.append(node.identity)   
        
    def sendSource2Sink(self, Ti):
        """
        一个周期内容事件：
        1）主要事件：源节点发送消息，并通过路径传输至基站
        更新的参数：
        1）节点剩余能量
        2）攻击者位置
        返回值：
        1）源节点是否被捕获
        2）源节点消息上传到基站的时间花费
        3）网络能耗
        """
        packetSize = 500  # 单次数据包大小 bit
        path = self.listPath[-1]
        # 路由时延 and 网络能耗
        delayTi = 0
        energyTi = 0
        u = -1
        for i,v in enumerate(path):
            if i == 0:
                u = v
            else:
                delayTi += self.delayModel(self.G, u, v)*packetSize
                tE, rE = self.pairEnergyModel(self.G.nodeList[u], self.G.nodeList[v])
                tE *= packetSize
                rE *= packetSize
                ec = (tE + rE)
                energyTi += ec
                self.G.nodeList[u].energy -= tE
                self.G.nodeList[v].energy += rE
                u = v
        # 攻击者移动
        self.attacker.tracebackNetwork(self.G)
        # 源节点是否被捕获
        flag = False
        if self.attacker.position.identity == self.source:
            flag = True
        else:
            flag = False
        return flag, delayTi, energyTi
    
    dfs_bestPath = []
    def searchDeepFirst(self, u, former, target, dist):
        """
        深度优先搜索
        """
        U = self.G.nodeList[u]
        for v in U.adj:
            V = self.G.nodeList[v]
            if V.level >= U.level or v in former:
                continue
            dist = dist + self.delayModel(self.G, u, v)
            former.append(v)
            # 每到一次 sink，更新一次 target 值
            if v == self.sink:
                # print former, "\nIt takes ", distVector[v], "ns"
                if dist < target:
                    target = dist
                    self.dfs_bestPath = former[:]
                    print target
                else:
                    pass
                return target
            else:
                target = self.searchDeepFirst(v, former, target, dist)
            # 回溯
            former.pop()
        return target
    
    def pathDeploy(self, Ti):
        """
        Phantom routing 部署 source 到 sink 的路径
        1）基于跳数的随机游走
        2）单路径路由策略
        
        return path (返回部署的路径)
        """
        # print "构建网络层次"       
        queue = Queue.Queue()
        visited = np.zeros(self.G.nodeNumber, dtype=np.int8)
        
        self.G.nodeList[self.sink].level = 0
        queue.put(self.sink)
        visited[self.sink] = 1
        # 计算节点与基站之间最短打跳数
        while not queue.empty():
            u = queue.get()
            visited[u] = 0
            U = self.G.nodeList[u]
            for v in U.adj:
                V = self.G.nodeList[v]
                if V.level == -1 or U.level + 1 < V.level:
                    V.level = U.level + 1
                    if visited[v] == 0:
                        queue.put(v)
                        visited[v] = 1
        
        # print "随机游走中..."
        # director: 1->up, 0->down
        path = [self.source]
        director = np.random.randint(low=0, high=2, dtype=np.int8)
        u = -1
        v = -1
        for i in range(1, self.Hwalk+1):
            if i == 0:
                u = self.source
            else:
                v = u
            U = self.G.nodeList[u]
            vs = []
            for v in U.adj:
                V = self.G.nodeList[v]
                if director == 1 and V.level >= U.level and u != v:
                    vs.append(V.identity)
                elif director == 0 and V.level <= U.level and u != v:
                    vs.append(V.identity)
            v = vs[np.random.randint(low=0, high=len(vs), dtype=np.int8)]
            path.append(v)
        # print "寻找最短路"
        self.distVector = np.ones(self.G.nodeNumber+2)*1e30
        self.distVector[path[-1]] = 0.
        _ = self.searchDeepFirst(path[-1], [path[-1]], 1e30, 0)
        # 随机游走 + 最短路
        path.extend(self.dfs_bestPath[1:])
        for v in range(self.G.nodeNumber):
            self.G.nodeList[v].weight = 0
        for v in path:
            self.G.nodeList[v].weight = 1
        return path
        
    def algPhantomRouting(self):
        """
        Phantom routing 过程
        """
        listDelay = []
        listEnergyConsumption = []
        safety = -1
        for Ti in range(1, self.Tmax+1):
            if Ti % 100 == 0:
                print Ti
            else:
                print Ti,
            path = self.pathDeploy(Ti)
            self.listPath.append(path)
            # 源节点发送消息给基站的事件
            flag, delayTi, energyTi = self.sendSource2Sink(Ti)
            # 保存每轮的记录
            listDelay.append(delayTi)
            listEnergyConsumption.append(energyTi)
            if flag:
                safety = Ti
                break
            elif Ti == self.Tmax:
                safety = self.Tmax
            else:
                pass
        return safety, listDelay, listEnergyConsumption
    def plotDelayandConsumption(self):
        """
        曲线：
        1）每轮的网络时延
        2）每轮的能耗
        """
        plt.figure(figsize=(5,6))
        plt.subplot(2,1,1)
        plt.semilogy(np.array(self.listDelay)/1e9)
        plt.title("Delay")
        plt.subplot(2,1,2)
        plt.semilogy(np.array(self.listEnergyConsumption)/self.G.nodeNumber)
        plt.title("Consumption")
        plt.show()
    def plotPath(self, Ti=-1):
        """
        数据传输路径的绘制
        Ti = 周期: -1, Ti>0
        """
        # 取最长的路径或第 Ti 轮的路径
        path =[]
        if Ti == -1:
            path = self.listPath[0]
            for p in self.listPath:
                if len(p) > len(path):
                    path = p
        else:
            path = self.listPath[Ti-1]
            
        ax = plt.figure(figsize=(5,5))
        temp_x = []
        temp_y = []
        for i in range(self.G.nodeNumber):
            if i in path:
                continue
            temp_x.append(self.G.nodeList[i].position[0])
            temp_y.append(self.G.nodeList[i].position[1])
        plt.plot(temp_x, temp_y, 'yp')
        # 传输路径
        u = -1
        for i,v in enumerate(path):
            if i == 0:
                u = v
                continue
            else:
                U = self.G.nodeList[u]
                V = self.G.nodeList[v]
                x = [U.position[0], V.position[0]]
                y = [U.position[1], V.position[1]]
                plt.plot(x, y, 'k')
                u = v
        # 传输的节点
        temp_x = []
        temp_y = []
        for i in path:
            temp_x.append(self.G.nodeList[i].position[0])
            temp_y.append(self.G.nodeList[i].position[1])
        plt.plot(temp_x, temp_y, 'rs')
        plt.axis("equal")
        plt.show()
    
    def phantomRouting(self):
        """
        Phantom Routing 主函数
        1）生成 sink 和 source
        2）网络连通性判定，直至网络连通
        3）phantom routing 及事件模拟
        其中，
        1）时延，单位 ns,纳秒
        2）能耗，单位 nj,纳焦
        """
        self.generateSINKandSOURCE()
        flag = self.isConnected  # 更新各节点邻居
        if not flag:
            print "网络不连通"
            return
        else:
            print "网络连通"
        self.deployAttacker(self.G.nodeList[self.sink]) # 部署攻击者位置
        self.safety, self.listDelay, self.listEnergyConsumption = self.algPhantomRouting()


# In[148]:

"""
调试中...深度优先搜索有问题：level 没有递减
"""
rp = routingPhantom(G=cpsNetwork(nodeNumber=500, areaLength=100, initEnergy=1e8, radius=20),                     Tmax=10, Hwalk=3)
# rp.display()
# rp.generateSINKandSOURCE()
# flag = rp.isConnected(rp.G, rp.sink) # 更新各节点邻居
# print flag
# rp.deployAttacker(rp.G.nodeList[rp.sink]) # 部署攻击者位置
# rp.algPhantomRouting()
rp.phantomRouting()
rp.plotDelayandConsumption()
rp.plotPath()


# In[149]:

P = rp.listPath[0]
for p in rp.listPath:
    if len(p) > len(P):
        P = p
print len(P), P
for i in P:
    print rp.G.nodeList[i].level,
print ""
print rp.attacker.trace


# In[22]:

fs = cpstopoFakeScheduling(nodeNumber=5000, areaLength=500, initEnergy=1e8, radius=20,                            Tmax=1000, c_capture=1e-3)
fs.fakeScheduling()


# ## 基于树结构的诱导路由技术

# ## 动态虚假源选择算法

# In[ ]:




# coding=utf-8
"""
Pandurang Kamat, Yanyong Zhang, Wade Trappe, Celal Ozturk. Enhancing source-location privacy in sensor network routing.
    In: Proceedings of 25th IEEE International Conference on Distributed Computing Systems (ICSCS'05). 2005	

# 对比算法
* phantom routing
* 基于树结构的诱导路由技术
* 动态虚假源选择算法

"""
from cpsNetwork import *
from cpsAttacker import *
from cpsNode import *
import datetime


class routingPhantom:
    """
    Phantom routing 技术
    --------------------
    用例：
    # 生成对象
    rp = routingPhantom(G=cpsNetwork(nodeNumber=1000, areaLength=100, initEnergy=1e8, radius=10), Tmax=100, Hwalk=3)
    # 主函数入口
    rp.phantomRouting()
    # 绘制时延与能耗曲线
    rp.plotDelayandConsumption()
    # 路径绘制
    rp.plotPath()
    --------------------
    变量成员：
    G = 网络
    Tmax = 最大轮数
    Hwalk = 随机步数
    sink = 基站编号
    sink_pos = 基站位置
    source = 源节点编号
    source_pos = 源节点位置
    attacker = 攻击者
    safety = 安全周期数
    listDelay = 每周期的传输时延
    listEnergyConsumption = 每周期网络能耗
    listPath = 每周期的传输路径
    --------------------
    方法成员：
    __init__(G, Tmax, Hwalk, sink_pos, source_pos) = 初始化函数
    display()
    generateSINKandSOURCE()
    deployAttacker(node)
    sendSource2Sink(Ti)
    searchDeepFirst(u, dist, former, target, bestPath)
    pathDeploy(Ti)
    algPhantomRouting()
    plotDelayandConsumption()
    plotPath(Ti)
    phantomRouting() = 主函数入口
    """

    def __init__(self, G=cpsNetwork(nodeNumber=10, areaLength=20, initEnergy=1e9, radius=50), Tmax=1000, Hwalk=30,
                 sink_pos=(0, 0), source_pos=(0.45 * 20, 0.4 * 20)):
        self.G = G
        self.Tmax = Tmax
        self.Hwalk = Hwalk
        self.sink = -1
        self.sink_pos = sink_pos
        self.source = -1
        self.source_pos = source_pos
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
        self.source = self.G.nodeNumber + 1
        sinkNode = cpsNode(identity=self.sink, position=self.sink_pos, energy=self.G.initEnergy * 100,
                           radius=self.G.radius, state='SINK', adj=set())
        sourceNode = cpsNode(identity=self.source, position=self.source_pos, energy=self.G.initEnergy * 100,
                             radius=self.G.radius, state='SOURCE', adj=set())
        num = self.G.nodeNumber
        self.G.addNode(sinkNode)
        self.G.addNode(sourceNode)
        if self.G.nodeNumber != num + 2:
            print "Error in the deployed sink and source."
            while len(self.G.nodeList) > num:
                self.G.nodeList.pop()
        else:
            for i in self.G.nodeList[self.sink].adj:
                self.G.nodeList[i].adj.add(self.sink)
            for i in self.G.nodeList[self.source].adj:
                    self.G.nodeList[i].adj.add(self.source)

    def deployAttacker(self, Node):
        """
        部署攻击者初始位置，位于 sink
        """
        self.attacker.initDeploy(Node)

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
        flag       = False    # 网络结束标志
        packetSize = 500 * 8  # 单次数据包大小 bit
        path       = self.listPath[-1]
        # 路由时延 and 网络能耗
        delayTi  = 0
        energyTi = 0
        u        = -1
        for i, v in enumerate(path):
            if i == 0:
                u = v
            else:
                delayTi += self.G.delayModelusingAdjMatrix(u,v) * packetSize
                u = v
            ec = self.G.energyModelusingAdjMatrix(self.G.nodeList[u]) * packetSize
            energyTi += ec
            self.G.nodeList[u].energy -= ec
            if not self.G.nodeList[u].isAlive():
                flag = True
        # 攻击者移动
        self.attacker.move(self.G)
        # 源节点是否被捕获
        if self.attacker.position.identity == self.source:
            flag = True
        # 每个周期执行的监听函数，用于获取网络信息
        self.getResult4AnalysisEachRound(Ti)
        return flag, delayTi, energyTi

    def getResult4AnalysisEachRound(self, Ti):
        """
		获取用于评估算法性能的方法
		:return: 性能指标
		"""
        if Ti == 1:
            self.result = []
        each_result = []
        # hop from sink to source
        each_result.append(len(self.listPath[-1]) - 1)
        # number of transmitted and broadcast fake messages
        each_result.append(0)
        self.result.append(each_result)

    def searchDeepFirst(self, u, dist, former, target, bestPath):
        """
        深度优先搜索:(前驱节点，当前长度，已走过的路径，目标值，最佳路径)
        """
        if target < np.inf:
            return target, bestPath
        U = self.G.nodeList[u]
        for v in U.adj:
            V = self.G.nodeList[v]
            if V.level >= U.level or v in former:
                continue
            dis = self.G.delayModelusingAdjMatrix(u, v)
            former.append(v)
            # 每到一次 sink，更新一次 target 值
            if v == self.sink:
                if dist + dis < target:
                    target = dist + dis
                    bestPath = former[:]
                else:
                    pass
                former.pop()  # 回溯
                return target, bestPath
            else:
                target, bestPath = self.searchDeepFirst(v, dist + dis, former, target, bestPath)
                former.pop()  # 回溯
        return target, bestPath

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
            for v in self.G.nodeList[u].adj:
                if self.G.nodeList[v].level == -1 or self.G.nodeList[u].level + 1 < self.G.nodeList[v].level:
                    self.G.nodeList[v].level = self.G.nodeList[u].level + 1
                    if visited[v] == 0:
                        queue.put(v)
                        visited[v] = 1

        # print "随机游走中..."
        # director: 1->up, 0->down
        path = [self.source]
        director = np.random.randint(low=0, high=2, dtype=np.int8)
        u = self.source
        v = -1
        for i in range(1, self.Hwalk + 1):
            U = self.G.nodeList[u]
            vs = []
            for v in U.adj:
                V = self.G.nodeList[v]
                if director == 1 and V.level >= U.level and v not in path:
                    vs.append(v)
                elif director == 0 and V.level <= U.level and v not in path:
                    vs.append(v)
            if len(vs):
                v = vs[np.random.randint(low=0, high=len(vs), dtype=np.int8)]
                path.append(v)
                u = v
            else:
                break
        # print "寻找最短路"
        target, bestPath = self.searchDeepFirst(path[-1], 0, [path[-1]], np.inf, [])
        # 随机游走 + 最短路  ########################################################################### weight to adjMatrix
        path.extend(bestPath[1:])
        for (u,v) in zip(path[:-1],path[1:]):
            self.G.adjacentMatrix[u,v] = 1
        return path

    def algPhantomRouting(self):
        """
        Phantom routing 过程
        """
        listDelay = []
        listEnergyConsumption = []
        safety = -1
        for Ti in range(1, self.Tmax + 1):
            if Ti % 100 == 0:
                print Ti
            elif Ti % 10 == 0:
                print Ti,
            self.G.initAdjMatrix()
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
        plt.figure(figsize=(5, 6))
        plt.subplot(2, 1, 1)
        plt.plot(np.array(self.listDelay) / 1e9)
        plt.title("Delay")
        plt.subplot(2, 1, 2)
        plt.plot(np.array(self.listEnergyConsumption))
        plt.title("Consumption")
        plt.show()

    def plotPath(self, Ti=-1):
        """
        数据传输路径的绘制
        Ti = 周期: -1, Ti>0
        """
        # 取最长的路径或第 Ti 轮的路径
        path = []
        if Ti == -1:
            path = self.listPath[0]
            for p in self.listPath:
                if len(p) > len(path):
                    path = p
        else:
            path = self.listPath[Ti - 1]

        ax = plt.figure(figsize=(5, 5))
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
        for i, v in enumerate(path):
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
        self.deployAttacker(self.G.nodeList[self.sink])  # 部署攻击者位置
        self.safety, self.listDelay, self.listEnergyConsumption = self.algPhantomRouting()
        print "\nThe safety is %d" % self.safety

if __name__ == "__main__":
    network = cpsNetwork(file_path='load_network/network.csv')
    print network.nodeNumber, network.areaLength

    rp = routingPhantom(G=network, Tmax=100, Hwalk=20, sink_pos=(-150,-150), source_pos=(150,150))
    rp.phantomRouting()
    rp.plotDelayandConsumption()
    rp.plotPath()
    rp.attacker.display()

    print np.array(rp.result)

    restEnergy = [rp.G.initEnergy - node.energy for node in rp.G.nodeList if
                  node.identity != rp.source and node.identity != rp.sink]
    # print restEnergy
    print max(restEnergy), np.mean(restEnergy), min(restEnergy), np.std(restEnergy)

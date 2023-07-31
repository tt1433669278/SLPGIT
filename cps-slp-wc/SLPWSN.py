# coding=utf-8
"""
# Source location privacy in Cyber-physical systems

- 论文算法设计
    - 骨干网络构建
    - 虚假消息广播
my_baseline
"""
import heapq
import math
import random

from matplotlib.patches import Wedge

from cpsNetwork import *
from cpsAttacker import *
from cpsNode import *
import networkx as nx


class SBSLP:
    """
	虚假源调度算法，包含两个阶段：
	1）骨干网络构建
	2）虚假消息调度
	--------------
	用例：
	# 生成对象
	fs = cpstopoFakeScheduling(G=cpsNetwork(nodeNumber=1000, areaLength=100, initEnergy=1e8, radius=10),\
							   Tmax=100, c_capture=1e-3)
	# 虚假源调度算法主函数
	fs.fakeScheduling()
	# 生成骨干网络
	fs.backbonePlot()
	# 结果绘制
	fs.plotDelayandConsumption()
	--------------
	变量成员：
	G          = 网络
	Tmax       = 最大周期数

	C_Capture  = 被捕获率阈值，1e-4
	C_alpha    = 超参数 alpha，0.5
	C_Beta     = 超参数 beta，0.5

	sink       = 基站编号
	sink_pos   = 基站位置，(0,0)
	source     = 源节点编号
	source_pos = 源节点位置，(0.45*20, 0.45*20)
	attacker   = 攻击者

	backbone   = 骨干网络
	safety     = 安全周期数
	listDelay  = 每周期的时延
	listEnergyConsumption = 每周期的传输能耗
	listFakeSource        = 每周期源节点序列，用节点编号表示
	--------------
	方法成员：
	__init__(G, Tmax, c_capture, c_alpha, c_beta, sink_pos, source_pos)
	display()
	generateSINKandSOURCE()
	deployAttacker(node)
	calculate2Distance(node1, node2)
	isConnected()
	searchDeepFirst(u, former, bestBackbone, target, likelihood, maxStep)
	generateBackbone()
	calculateFakeSource(node, Ti)
	delayModel(u, v)
	consumeEnergyModel(node)
	sendSource2Sink(Ti)
	scheduingFakeMessages()
	resultPlot()
	backbonePlot()
	fakeScheduling()     = 虚假源调度主算法入口
	"""

    def __init__(self, G=cpsNetwork(nodeNumber=10, areaLength=20, initEnergy=1e6, radius=10), Tmax=1000, c_capture=1e-4,
                 c_alpha=0.5, c_beta=0.5, sink_pos=(0, 0), source_pos=(0.45 * 20, 0.4 * 20)):
        self.t_point = 1
        self.path = []
        self.dypath = []
        self.sum_path = []
        self.G = G
        self.Tmax = Tmax

        self.C_Capture = c_capture
        self.C_Alpha = c_alpha
        self.C_Beta = c_beta

        self.sink = -1
        self.sink_pos = sink_pos
        self.source = -1
        self.source_pos = source_pos
        self.attacker = cpsAttacker()

        self.safety = -1  # 网络安全周期
        self.listDelay = []  # 每个周期打网络时延
        self.listEnergyConsumption = []  # 每个周期的网络能耗
        self.listFakeSource = []  # 每个周期的虚假源部署情况
        self.open_set = []
        self.closed_set = []
        self.min_node = None
        self.nodenum = []

        self.brw_path = []
        self.lpw_path = []
        self.lpw2_path = []
        self.spr_path = []
        self.brw_next_node = None
        self.lpw_next_node = None

    def display(self):
        print "节点总数：", self.G.nodeNumber
        print "正方形区域边长：", self.G.areaLength
        print "节点初始能量：", self.G.initEnergy
        print "最大周期数：", self.Tmax
        print "节点通信半径：", self.G.radius
        print "捕获率阈值：", self.C_Capture
        print "参数 alpha：", self.C_Alpha
        print "参数 beta：", self.C_Beta
        print "sink 编号：", self.sink
        print "source 编号：", self.source

    def generateSINKandSOURCE(self):
        "部署 sink 和 source"
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
            for i in self.G.nodeList[self.sink].adj:  # 把sik点和source点加到邻居列表中
                self.G.nodeList[i].adj.add(self.sink)
            for i in self.G.nodeList[self.source].adj:
                self.G.nodeList[i].adj.add(self.source)

    def deployAttacker(self, Node):
        """
		部署攻击者初始位置，位于 sink
		"""
        self.attacker.initDeploy(Node)

    def generateNetworkLevel(self):
        print "构建网络层次"
        queue = Queue.Queue()
        visited = np.zeros(self.G.nodeNumber, dtype=np.int8)

        self.G.nodeList[self.sink].level = 0
        queue.put(self.sink)
        visited[self.sink] = 1
        # 计算节点与基站之间最短的跳数
        while not queue.empty():
            u = queue.get()
            visited[u] = 0
            for v in self.G.nodeList[u].adj:
                if self.G.nodeList[v].level == -1 or self.G.nodeList[u].level + 1 < self.G.nodeList[v].level:
                    self.G.nodeList[v].level = self.G.nodeList[u].level + 1
                    if visited[v] == 0:
                        queue.put(v)
                        visited[v] = 1

    def getResult4AnalysisEachRound(self, Ti):
        """
        获取用于评估算法性能的方法
        :return: 性能指标
        """
        if Ti == 1:
            self.result = []
        each_result = []
        # hop from sink to source
        each_result.append(len(self.sum_path) - 1)
        # number of transmitted and broadcast fake messages对邻接矩阵的每一行进行求最大值的操作，得到一个包含每个节点传输和接收的边数的数组
        each_result.append(np.max(self.G.adjacentMatrix, axis=1).sum() - len(self.sum_path) + 1)
        self.result.append(each_result)

    def brwphase(self, node, brw_path, h1):
        self.brw_path = brw_path
        forwarding_set = []
        brw_next_node = None
        # while True:
        for i in self.G.nodeList[node].adj:
            if self.G.nodeList[i].level >= self.G.nodeList[node].level and i not in self.brw_path:
                forwarding_set.append(i)
        if forwarding_set:
            brw_next_node = random.choice(forwarding_set)
            self.brw_path.append(brw_next_node)
            h1 -= 1
            if h1 == 0:
                self.brw_next_node = brw_next_node
                print 'brw阶段路径：', self.brw_path
                return self.brw_path, self.brw_next_node
            self.brwphase(brw_next_node, self.brw_path, h1)
        else:
            print "forwarding_set is no"

        return self.brw_path, self.brw_next_node

    def lpwphase(self, node):
        """
        action: 1-up , 2-down , 3-left , 4-right
        """
        self.lpw_path.append(node)
        a = 0.5
        random_number = random.randint(0, 1)
        if random_number < a:
            action = random.choice([3, 4])
        else:
            action = random.choice([1, 2])
        if action == 1:
            print "选择上"
        elif action == 2:
            print "选择下"
        elif action == 3:
            print "选择左"
        else:
            print "选择右"
        if action == 1 or action == 2:
            print "第一阶段选择上下"
            h2 = random.randint(0, int(network.areaLength / self.G.radius))
            next_action, lpw1_path, nex_node, h = self.lpwone(node, [node], h2, action)
            if h == 0:
                print '随机步数为：', h2, 'lpw第一阶段完整路径：', lpw1_path, h
            else:
                print '随机步数为：', h2, 'lpw第一阶段不完整路径：', lpw1_path, '最后点：', nex_node
            if next_action == 3 or next_action == 4:
                print "第二阶段选择左右"
                h2 = random.randint(0, int(network.areaLength / self.G.radius))
                p, lpw2_path, nex_node, h = self.lpwtwo(node, [nex_node], h2, action)
                if h == 0:
                    print '随机步数为：', h2, 'lpw第二阶段完整路径：', lpw2_path, h
                else:
                    print '随机步数为：', h2, 'lpw第二阶段不完整路径：', lpw2_path, '最后点：', nex_node, h
        if action == 3 or action == 4:
            print "第一阶段选择左右"
            h2 = random.randint(0, int(network.areaLength / self.G.radius))
            next_action, lpw1_path, nex_node, h = self.lpwtwo(node, [node], h2, action)
            if h == 0:
                print '随机步数为：', h2, 'lpw第一阶段完整路径：', lpw1_path, h, '最后点：', nex_node
            else:
                print '随机步数为：', h2, 'lpw第一阶段不完整路径：', lpw1_path, '最后点：', nex_node, h
            if next_action == 1 or next_action == 2:
                print "第二阶段选择上下"
                h2 = random.randint(0, int(network.areaLength / self.G.radius))
                p, lpw2_path, nex_node2, h = self.lpwone(node, [nex_node], h2, action)
                if h == 0:
                    print '随机步数为：', h2, 'lpw第二阶段完整路径：', lpw2_path, h
                else:
                    print '随机步数为：', h2, 'lpw第二阶段不完整路径：', lpw2_path, '最后点：', nex_node2, h


        self.lpw_next_node = nex_node
        print 'lpw阶段完整路径： ', self.lpw_path, '最后点为：', self.lpw_next_node
        self.spr_path.append(self.lpw_next_node)
        self.sum_path.append(self.lpw_path)
        return self.lpw_path, self.lpw_next_node

    def lpwone(self, node, path, h2, action):
        lpw1_path = path
        # self.lpw_path = [node]
        slect_node = []
        lpw_next_node = None
        if node:
            for a in self.G.nodeList[node].adj:
                if a not in self.lpw_path and a not in self.brw_path:
                    vec_x = self.G.nodeList[a].position[0] - self.G.nodeList[node].position[0]
                    vec_y = self.G.nodeList[a].position[1] - self.G.nodeList[node].position[1]
                    angle = math.degrees(math.atan2(vec_y, vec_x))
                    if action == 1 and 0 < angle < 180:
                        slect_node.append(a)
                    if action == 2 and -180 < angle < 0:
                        slect_node.append(a)
            if slect_node:
                lpw_next_node = random.choice(slect_node)
                lpw1_path.append(lpw_next_node)
                self.lpw_path.append(lpw_next_node)
                h2 -= 1
                if h2 == 0:
                    next_action = random.choice([3, 4])
                    return next_action, lpw1_path, lpw_next_node, h2
                return self.lpwone(lpw_next_node, lpw1_path, h2, action)
            else:
                next_action = random.choice([3, 4])
                return next_action, lpw1_path, node, h2
        else:
            print "node is NONE"

    def lpwtwo(self, node, path, h2, action):
        lpw2_path = path
        slect_node = []
        lpw_next_node = None
        for a in self.G.nodeList[node].adj:
            if a not in lpw2_path and a not in self.brw_path:
                vec_x = self.G.nodeList[a].position[0] - self.G.nodeList[node].position[0]
                vec_y = self.G.nodeList[a].position[1] - self.G.nodeList[node].position[1]
                angle = math.degrees(math.atan2(vec_y, vec_x))
                if action == 3 and 90 < angle < 180 or -180 < angle < -90:
                    slect_node.append(a)
                if action == 4 and 0 < angle < 90 or -90 < angle < 0:
                    slect_node.append(a)

        if slect_node:
            lpw_next_node = random.choice(slect_node)
            lpw2_path.append(lpw_next_node)
            self.lpw_path.append(lpw_next_node)
            h2 -= 1
            if h2 == 0:
                next_action = random.choice([1, 2])
                return next_action, lpw2_path, lpw_next_node, h2
            return self.lpwtwo(lpw_next_node, lpw2_path, h2, action)
        else:
            next_action = random.choice([1, 2])
            return next_action, lpw2_path, node, h2

    def sprphase(self, node):
        num_node = []
        for i in self.G.nodeList[node].adj:
            if self.G.nodeList[i].level < self.G.nodeList[node].level:
                num_node.append(i)
        sle_node = random.choice(num_node)
        self.spr_path.append(sle_node)
        # self.sprphase(sle_node)
        if sle_node != self.sink:
            self.sprphase(sle_node)
        else:
            print "spr阶段结束，路径为：", self.spr_path
            self.sum_path.append(self.spr_path)
            print "总路线为：", self.sum_path

    def pathplot(self):
        ax = plt.figure(figsize=(20, 20))
        nom_x = []
        nom_y = []
        for i in range(self.G.nodeNumber):
            if i in self.brw_path and self.lpw_path and self.spr_path:
                continue
            nom_x.append(self.G.nodeList[i].position[0])
            nom_y.append(self.G.nodeList[i].position[1])
        plt.plot(nom_x, nom_y, 'ko')
        # brw_path骨干
        u = -1
        for i, v in enumerate(self.brw_path):
            if i == 0:
                u = v
                continue
            else:
                U = self.G.nodeList[u]
                V = self.G.nodeList[v]
                x = [U.position[0], V.position[0]]
                y = [U.position[1], V.position[1]]
                plt.plot(x, y, 'k')  # 绘制两点之间连线
                u = v
        # lpw_path骨干
        m = -1
        for i, v in enumerate(self.lpw_path):
            if i == 0:
                m = v
                continue
            else:
                U = self.G.nodeList[m]
                V = self.G.nodeList[v]
                x = [U.position[0], V.position[0]]
                y = [U.position[1], V.position[1]]
                plt.plot(x, y, 'k')  # 绘制两点之间连线
                m = v
        # spr_path骨干
        r = -1
        for i, v in enumerate(self.spr_path):
            if i == 0:
                r = v
                continue
            else:
                U = self.G.nodeList[r]
                V = self.G.nodeList[v]
                x = [U.position[0], V.position[0]]
                y = [U.position[1], V.position[1]]
                plt.plot(x, y, 'k')  # 绘制两点之间连线
                r = v
        # brw_path节点
        temp_x = []
        temp_y = []
        for i in self.brw_path:
            temp_x.append(self.G.nodeList[i].position[0])
            temp_y.append(self.G.nodeList[i].position[1])
        plt.plot(temp_x, temp_y, 'ro')  # ro红圆ko黑圆wo白圆rs红方
        # lpw_path节点
        mhr_x = []
        mhr_y = []
        for i in self.lpw_path:
            mhr_x.append(self.G.nodeList[i].position[0])
            mhr_y.append(self.G.nodeList[i].position[1])
        plt.plot(mhr_x, mhr_y, 'mo')
        # spr_path节点
        spr_x = []
        spr_y = []
        for i in self.spr_path:
            spr_x.append(self.G.nodeList[i].position[0])
            spr_y.append(self.G.nodeList[i].position[1])
        plt.plot(spr_x, spr_y, 'bo')

        plt.axis("equal")
        plt.show()

    def sendSource2Sink(self, Ti):
        """
        一个周期内容事件：
        1）主要事件：源节点（包含分支上的虚假源）发送消息，并通过路径传输至基站
        输入参数：
        1）网络的链接矩阵 G.adjacentMatrix
        更新的参数：
        1）节点剩余能量
        2）攻击者位置
        返回值：
        1）源节点是否被捕获
        2）源节点消息上传到基站的时间花费
        3）网络能耗
        """

        packetSize = 500 * 8  # 单次数据包大小 bit
        # 路由时延 and 网络能耗
        flag = False
        delayTi = 0
        energyTi = 0
        # 延迟
        u = -1
        for i, v in enumerate(self.brw_path):
            if i == 0:
                u = v
            else:
                delayTi += self.G.delayModelusingAdjMatrix(u, v) * packetSize
                u = v
        for i, v in enumerate(self.lpw_path):
            if i == 0:
                u = v
            else:
                delayTi += self.G.delayModelusingAdjMatrix(u, v) * packetSize
                u = v
        for i, v in enumerate(self.spr_path):
            if i == 0:
                u = v
            else:
                delayTi += self.G.delayModelusingAdjMatrix(u, v) * packetSize
                u = v
        # 网络能耗
        for node in self.G.nodeList:
            ec = self.G.energyModelusingAdjMatrix(node) * packetSize
            energyTi += ec
            self.G.nodeList[node.identity].energy -= ec
            if not self.G.nodeList[node.identity].isAlive():
                flag = True
        # 攻击者移动
        self.attacker.move(self.G)
        # 源节点是否被捕获
        if self.attacker.position.identity == self.source:
            print " "
            print "BE CAPTURE"
            flag = True
        # 每个周期执行的监听函数，用于获取网络信息
        self.getResult4AnalysisEachRound(Ti)
        return flag, delayTi, energyTi

    def updateAdjMatrix(self):
        self.G.initAdjMatrix()
        for (u, v) in zip(self.brw_path[:-1], self.brw_path[1:]):
            self.G.adjacentMatrix[u, v] = 1
        for (q, w) in zip(self.lpw_path[:-1], self.lpw_path[1:]):
            self.G.adjacentMatrix[q, w] = 1
        for (e, r) in zip(self.spr_path[:-1], self.spr_path[1:]):
            self.G.adjacentMatrix[e, r] = 1

    def routing(self):
        """
        过程
        """
        listDelay = []
        listEnergyConsumption = []
        safety = -1
        for Ti in range(1, self.Tmax + 1):
            if Ti % 100 == 0:
                print Ti
            elif Ti % 10 == 0:
                print Ti
            # start = datetime.datetime.now()
            print " "
            print "the", Ti, "step"
            self.spr_path = []
            self.lpw_path = []
            self.sum_path = []
            self.brwphase(self.source, [self.source], 5)
            self.sum_path.append(self.brw_path)
            self.lpwphase(self.brw_next_node)
            self.sprphase(self.lpw_next_node)
            # 源节点发送消息给基站的事件
            flag, delayTi, energyTi = self.sendSource2Sink(Ti)
            self.updateAdjMatrix()
            # 保存每轮的记录
            listDelay.append(delayTi)
            listEnergyConsumption.append(energyTi)
            #             end = datetime.datetime.now()
            #             print '\nIt costs', end-start
            if Ti == self.Tmax:
                safety = Ti
                print "到达阈值"
                break
            if flag:
                safety = Ti
                print "被捕获"
                break
        return safety, listDelay, listEnergyConsumption

    def plotDelayandConsumption(self):
        """
        曲线：
        1）每轮的网络时延
        2）每轮的能耗
        """
        plt.figure(figsize=(15, 6))
        plt.plot(np.array(self.listEnergyConsumption))
        plt.title("Consumption")
        plt.show()

        plt.figure(figsize=(15, 6))
        plt.plot(np.array(self.listDelay) / 1e9)
        plt.title("Delay")
        plt.show()

    def section(self):
        sum_delay = 0
        self.generateSINKandSOURCE()
        self.generateNetworkLevel()
        self.deployAttacker(self.G.nodeList[self.sink])  # 部署攻击者位置
        # self.brwphase(self.source, [self.source], 5)
        # self.sum_path.append(self.brw_path)
        # self.lpwphase(self.brw_next_node)
        # self.sprphase(self.lpw_next_node)

        self.safety, self.listDelay, self.listEnergyConsumption = self.routing()
        self.pathplot()
        for i in range(len(self.listDelay)):
            sum_delay += self.listDelay[i]
            mean_delay = sum_delay / (i + 1)
        print "\nThe safety is", self.safety, "\nThe every listDelay is", self.listDelay, "\nThe SumDelay is", sum_delay, "\nThe MeanDelay is", mean_delay


# def test(self):
#     print self.attacker.position.identity


if __name__ == '__main__':
    # network = cpsNetwork(file_path='load_network/temp_network.csv')
    network = cpsNetwork(file_path='load_network/network.csv')
    print '网络规模：', network.nodeNumber, network.areaLength

    sb = SBSLP(G=network,
               Tmax=2000, c_capture=1e-40, c_alpha=0.02, c_beta=0.6,
               sink_pos=(-200, -200), source_pos=(200, 200))
    sb.section()

    # print np.array(fs.result)

    restEnergy = [sb.G.initEnergy - node.energy for node in sb.G.nodeList if
                  node.identity != sb.source and node.identity != sb.sink]
    # print restEnergy
    print "\nThe maxrestEnergy is", max(restEnergy), "\nThe neanrestEnergy is", np.mean(restEnergy), "\nThe minrestEnergy is", min(restEnergy), "\nThe stdrestEnergy is", np.std(restEnergy)
    # 最大值、平均值、最小值和标准差
    sb.attacker.display()

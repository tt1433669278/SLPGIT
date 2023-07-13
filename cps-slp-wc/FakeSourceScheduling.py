# coding=utf-8
"""
# Source location privacy in Cyber-physical systems

- 论文算法设计
    - 骨干网络构建
    - 虚假消息广播

"""
from cpsNetwork import *
from cpsAttacker import *
from cpsNode import *


class cpstopoFakeScheduling:
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

        self.backbone = []
        self.safety = -1  # 网络安全周期
        self.listDelay = []  # 每个周期打网络时延
        self.listEnergyConsumption = []  # 每个周期的网络能耗
        self.listFakeSource = []  # 每个周期的虚假源部署情况

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

    def searchDeepFirst(self, u, former, bestBackbone, target, likelihood, maxStep):
        """
		深度优先搜索：(前驱节点，已走过的路径，最佳路径，目标值，当前似然，最大步数)
		寻找所有长度为 maxStep 的路径，满足 self.C_Capture
		"""
        if target < np.log10(1.0):
            return target, bestBackbone
        vs = [v for v in self.G.nodeList[u].adj if v not in former
              and self.G.nodeList[v].level <= self.G.nodeList[u].level
              and len(former) + self.G.nodeList[v].level <= maxStep]
        np.random.shuffle(vs)  # 进行随机打乱顺序
        for v in vs:  # self.G.nodeList[u].adj:
            if v not in former and self.G.nodeList[v].level <= self.G.nodeList[u].level \
                    and len(former) + self.G.nodeList[v].level <= maxStep:  # 长度限制
                diff = len(self.G.nodeList[v].adj)
                former.append(v)
                # 每到一次 sink，更新一次 target 值
                if v == self.sink:
                    if likelihood - np.log10(diff + 1.0) < target:
                        target = likelihood - np.log10(diff + 1.0)
                        bestBackbone = former[:]
                    # 回溯
                    former.pop()
                    return target, bestBackbone
                else:
                    target, bestBackbone = self.searchDeepFirst(v, former, bestBackbone,
                                                                target, likelihood - np.log10(diff + 1.0), maxStep)
                    # 回溯
                    former.pop()
                    # 当 target 被更新以后（即已经到达过 sink），且满足要求，搜索结束
                    if target < np.log10(1.0):
                        return target, bestBackbone
        return target, bestBackbone

    def generateBackbone(self):
        """
		骨干网络构建算法
		"""
        print "构建骨干网络..."
        queue = Queue.Queue()  # 创建一个队列对象，用于辅助进行广度优先搜索
        visited = np.zeros(self.G.nodeNumber, dtype=np.int8)  # 初始化一个数组 visited 用于标记节点是否被访问过。

        self.G.nodeList[self.sink].level = 0  # 设置最短跳数
        queue.put(self.sink)
        visited[self.sink] = 1
        # 计算节点与基站之间最短打跳数
        while not queue.empty():  # 检查队列是否为空 设置每个节点的跳数 ！！！！！！！！！！！！！！！！！！！！
            u = queue.get()
            visited[u] = 0  # 上面给1又给0？
            for v in self.G.nodeList[u].adj:
                if self.G.nodeList[v].level == -1 or self.G.nodeList[u].level + 1 < self.G.nodeList[v].level:
                    self.G.nodeList[v].level = self.G.nodeList[u].level + 1
                    if visited[v] == 0:
                        queue.put(v)
                        visited[v] = 1

        sourceLikelihood = np.log10(10.)  # 被捕获似然的目标值 log(sourceLikelihood)
        sourceLevel = self.G.nodeList[self.source].level + 1  # 记录源节点与基站的跳数
        bestBackbone = []  # 传输路径

        # 迭代搜索满足被捕获似然打的骨干网络
        while sourceLikelihood > np.log10(self.C_Capture):
            # 评估源节点被捕获似然（深搜直至发现满意骨干网络）
            target, bestBackbone = self.searchDeepFirst(self.source, [self.source], [],
                                                        np.log10(1.0), np.log10(1.0), sourceLevel)
            # 记录似然值
            sourceLikelihood = target
            if sourceLikelihood == -1:  # 未找到 sink 节点
                sourceLikelihood = np.log10(1.)
            else:
                pass
            # 迭代更新跳数长度
            sourceLevel += 1
            if sourceLevel >= self.G.nodeNumber:
                break
        # 记录骨干网络
        # print "The sourceLikelihood is", sourceLikelihood
        if sourceLikelihood < self.C_Capture:
            self.backbone = bestBackbone
        print "骨干网络：", bestBackbone

    def calculateFakeSource(self, node, Ti):
        """
		计算节点间通信的概率，并返回 True/False 值
		PS: sink, source, backbone 不会成为 fake source
		"""
        if node.state == "SINK" or node.state == "SOURCE":
            return False
        B = set(self.backbone)
        if node.identity in B:
            return False

        # I_v_i^B
        numB = 0
        for v in node.adj:  # 如果邻居点有骨干节点的+1
            if v in B:
                numB += 1
        # rank_v_i^E
        rankEV = -1  # 节点能耗排名，周围节点少说明在稀疏地段需要通过更长的跳数来传输数据会消耗更多的能量
        if Ti == 1:
            rankEV = 1. / len(node.adj)
        else:
            rank = 1
            for v in node.adj:
                V = self.G.nodeList[v]
                if V.energy > node.energy:  # 计算节点能耗排名，能耗多的排名靠前
                    rank += 1
            rankEV = rank * 1. / len(node.adj)
        # I_v_i^C  循环统计了节点node的邻居节点中具有连接强度为1的节点的数量
        numC = 0
        if Ti == 1:
            numC = 0
        else:
            for v in node.adj:
                V = self.G.nodeList[v]
                if V.weight == 1:
                    numC += 1
        # C(p_i^T(l-1))
        CP = 0
        if Ti == 1:
            CP = 0
        else:
            CP = node.weight
        # p_i
        numI = len(node.adj)
        p_i_z = self.C_Alpha * np.exp(numB * 1. / numI)  # 分子
        p_i_m = self.C_Beta * np.exp(1. - rankEV * 1. / numI) + (1 - self.C_Beta) * np.exp(CP - numC * 1. / numI)  # 分母
        p_i = p_i_z / p_i_m  # 概率阈值
        # 是否广播
        RAND = np.random.rand()
        if RAND < p_i:
            return True
        else:
            return False

    def sendSource2Sink(self, Ti):
        """
		一个周期内容事件：
		1）主要事件：源节点发送消息，并通过骨干网络传输至基站
		2）攻击者移动模型
		3）虚假源广播
		更新的参数：
		1）节点剩余能量
		2）攻击者位置
		返回值：
		1）源节点是否被捕获
		2）源节点消息上传到基站的时间花费
		3）网络能耗
		"""
        flag = False
        packetSize = 500 * 8  # 单次数据包大小 bit
        # 路由时延
        delayTi = 0
        energyTi = 0
        for (u, v) in zip(self.backbone[:-1], self.backbone[1:]):
            self.G.adjacentMatrix[u, v] = 1
            delayTi += self.G.delayModelusingAdjMatrix(u, v) * packetSize
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
            flag = True
        # 每个周期执行的监听函数，用于获取网络信息
        self.getResult4AnalysisEachRound(Ti)
        return flag, delayTi, energyTi

    def updateAdjMatrix(self):
        """
		更新通信矩阵
		fake source broadcast a message to neighbors:通过更新通信矩阵，将虚假消息源节点与其邻居节点之间的通信关系标记为1
			self.G.adjacentMatrix['FAKE', neighbors] = 1  模拟虚假消息的传播过程。
		"""
        self.G.initAdjMatrix()
        for node in self.G.nodeList:
            if node.state == 'FAKE':
                for v in node.adj:
                    self.G.adjacentMatrix[node.identity, v] = 1

    def scheduingFakeMessages(self):
        """
		虚假消息调度
		"""
        listDelay = []
        listEnergyConsumption = []
        safety = -1
        for Ti in range(1, self.Tmax + 1):
            if Ti % 100 == 0:
                print Ti
            elif Ti % 10 == 0:
                print Ti,
            # fake source scheduling
            for node in self.G.nodeList:
                if node.identity != self.sink and node.identity != self.source:
                    self.G.nodeList[node.identity].state = 'FAKE' if self.calculateFakeSource(node, Ti) else 'NORMAL'
            # update 节点权重，1：fake，0：not fake
            self.updateAdjMatrix()
            self.listFakeSource.append([node.identity for node in self.G.nodeList if node.state == 'FAKE'])
            # 源节点发送消息给基站的事件
            flag, delayTi, energyTi = self.sendSource2Sink(Ti)
            # 保存每轮的记录
            listDelay.append(delayTi)
            listEnergyConsumption.append(energyTi)
            if flag or Ti == self.Tmax:
                safety = Ti
                break
        return safety, listDelay, listEnergyConsumption

    def plotDelayandConsumption(self):
        """
		曲线：
		1）每轮的网络时延
		2）每轮的能耗
		"""
        plt.figure(figsize=(15, 6))
        plt.subplot(211)  # 行列
        plt.plot(np.array(self.listDelay) / 1e9)
        plt.title("Delay")
        plt.subplot(212)
        plt.plot(np.array(self.listEnergyConsumption))
        plt.title("Consumption")
        plt.show()

    def backbonePlot(self):
        """
		骨干网络绘制
		"""
        ax = plt.figure(figsize=(10, 10))
        temp_x = []
        temp_y = []
        for i in range(self.G.nodeNumber):
            if i in self.backbone:
                continue
            temp_x.append(self.G.nodeList[i].position[0])
            temp_y.append(self.G.nodeList[i].position[1])
        plt.plot(temp_x, temp_y, 'ko')
        # 骨干网络
        u = -1
        for i, v in enumerate(self.backbone):
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
        # 骨干节点
        temp_x = []
        temp_y = []
        for i in self.backbone:
            temp_x.append(self.G.nodeList[i].position[0])
            temp_y.append(self.G.nodeList[i].position[1])
        plt.plot(temp_x, temp_y, 'ro')  # ro红圆ko黑圆wo白圆rs红方
        plt.axis("equal")
        plt.show()

    def getResult4AnalysisEachRound(self, Ti):
        """
		获取用于评估算法性能的方法
		:return: 性能指标
		"""
        if Ti == 1:
            self.result = []
        each_result = []
        # hop from sink to source
        each_result.append(len(self.backbone) - 1)
        # number of transmitted and broadcast fake messages对邻接矩阵的每一行进行求最大值的操作，得到一个包含每个节点传输和接收的边数的数组
        each_result.append(np.max(self.G.adjacentMatrix, axis=1).sum() - len(self.backbone) + 1)
        self.result.append(each_result)

    def fakeScheduling(self):
        """
		虚假源调度算法的主函数
		1）生成 sink 和 source
		2）网络连通性判定，直至网络连通
		3）骨干网络构建
		4）虚假源调度及事件模拟
		其中，
		1）时延，单位 ns,纳秒
		2）能耗，单位 nj,纳焦
		"""
        sum_delay = 0
        self.generateSINKandSOURCE()
        self.deployAttacker(self.G.nodeList[self.sink])  # 部署攻击者位置
        self.generateBackbone()  # 生成骨干网络
        self.safety, self.listDelay, self.listEnergyConsumption = self.scheduingFakeMessages()  # 虚假源调度与网络路由事件
        for i in range(len(self.listDelay)):
            sum_delay += self.listDelay[i]
            mean_delay = sum_delay / (i + 1)
        print "\nThe safety is", self.safety, "\nThe every listDelay is", self.listDelay, "\nThe SumDelay is", sum_delay, "\nThe MeanDelay is", mean_delay


# def test(self):
#     print self.attacker.position.identity


if __name__ == '__main__':
    network = cpsNetwork(file_path='load_network/network.csv')
    print '网络规模：', network.nodeNumber, network.areaLength

    fs = cpstopoFakeScheduling(G=network,
                               Tmax=2000, c_capture=1e-40, c_alpha=0.02, c_beta=0.6,
                               sink_pos=(-200, -200), source_pos=(200, 200))
    fs.fakeScheduling()

    # print np.array(fs.result)

    restEnergy = [fs.G.initEnergy - node.energy for node in fs.G.nodeList if
                  node.identity != fs.source and node.identity != fs.sink]
    # print restEnergy
    print "\nThe maxrestEnergy is", max(restEnergy), "\nThe neanrestEnergy is", np.mean(restEnergy), "\nThe minrestEnergy is", min(restEnergy), "\nThe stdrestEnergy is", np.std(restEnergy)

    # result_dict = dict()
    # for ca in np.linspace(0.2, 1, 5):
    # 	for cb in np.linspace(0, 1, 6):
    # 		fs.initModel()
    # 		fs.c_alpha = ca
    # 		fs.c_beta = cb
    # 		print 'Alpha = %.2f, Beta = %.2f' % (ca, cb)
    # 		experiment_energy = []
    # 		experiment_delay = []
    # 		experiment_safety = []
    # 		if flag:
    # 			for i in range(10):
    # 				print '\nExperiment %d' % (i + 1)
    # 				fs.deployAttacker(fs.G.nodeList[fs.sink])  # 部署攻击者位置
    # 				fs.generateBackbone()  # 生成骨干网络
    # 				safety, listDelay, listEnergyConsumption = fs.scheduingFakeMessages()  # 虚假源调度与网络路由事件
    # 				# 输出的指标
    # 				fs.safety = safety
    # 				fs.listDelay = listDelay
    # 				fs.listEnergyConsumption = listEnergyConsumption
    # 				print "\nThe safety is", fs.safety
    # 				experiment_energy.append(max([1e8 - node.energy for node in fs.G.nodeList \
    # 											  if node.identity != fs.sink]))
    # 				experiment_delay.append(max(fs.listDelay))
    # 				experiment_safety.append(fs.safety)
    # 			result_dict[(ca, cb)] = (np.mean(experiment_energy), np.mean(experiment_delay), np.mean(experiment_safety))

    fs.backbonePlot()
    fs.plotDelayandConsumption()
    # 每轮的虚假源节点数量
    a = [len(x) for x in fs.listFakeSource]
    print 'a', a
    plt.figure(figsize=(15, 3))
    plt.plot(a)
    plt.ylabel('The number of fake source')
    plt.show()
    fs.attacker.display()

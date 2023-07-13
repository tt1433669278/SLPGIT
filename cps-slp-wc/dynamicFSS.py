# coding=utf-8
"""
Bradbury M, Leeke M, Jhumka A. A dynamic fake source algorithm for source location privacy in wireless sensor networks[C].
	In: Proceedings of 2015 IEEE in Trustcom/BigDataSE/ISPA. 2015
"""
from cpsNetwork import *
from cpsAttacker import *
from cpsNode import *
import Queue
import matplotlib.pyplot as plt

class dynamicFSS:

	def __init__(self, G=cpsNetwork(nodeNumber=10, areaLength=20, initEnergy=1e6, radius=50),
                 Tmax=1000, sink_pos=(0,0), source_pos=(250, 250)):
		"""
		实例初始化
		:param G: 
		:param Tmax: 
		:param sink_pos: 
		:param source_pos: 
		"""
		if G.__class__ == 'Network from csv file'.__class__:
			self.G = cpsNetwork()
			self.G.read_csv(G)
		else:
			self.G = G

		self.Tmax = Tmax

		self.sink = -1
		self.sink_pos = sink_pos
		self.source = -1
		self.source_pos = source_pos

		self.attacker = cpsAttacker()

		self.safety = -1
		self.listDelay = []
		self.listEnergyConsumption = []

		self.path = []

	def display(self):
		print "节点总数：", self.G.nodeNumber
		print "正方形区域边长：", self.G.areaLength
		print "节点初始能量：", self.G.initEnergy
		print "节点通信半径：", self.G.radius
		print "sink：", self.sink, self.sink_pos
		print "source：", self.source, self.source_pos

	def display_results(self):
		print "安全周期数：", self.safety
		print "每一个周期的最大时延：", max(self.listDelay) if self.listDelay else 0
		print "每一个周期的最大能耗：", max(self.listEnergyConsumption) if self.listEnergyConsumption else 0
		id, maxEC = -1, 0
		for Node in self.G.nodeList:
			ec = min(self.G.initEnergy-Node.energy, self.G.initEnergy)
			if maxEC < ec:
				maxEC = ec
				id = Node.identity
		if id != -1:
			print "能耗最大的节点：", id, maxEC
		else:
			print "一切都好好的，节点的能量都满满的"

	def generateSINKandSOURCE(self):
		"部署 sink 和 source"
		self.sink = self.G.nodeNumber
		self.source = self.G.nodeNumber+1
		sinkNode = cpsNode(identity=self.sink, position=self.sink_pos, energy=self.G.initEnergy*100,
						   radius=self.G.radius, state='SINK', adj=set())
		sourceNode = cpsNode(identity=self.source, position=self.source_pos, energy=self.G.initEnergy*100,
							 radius=self.G.radius, state='SOURCE', adj=set())
		num = self.G.nodeNumber
		self.G.addNode(sinkNode)
		self.G.addNode(sourceNode)
		if self.G.nodeNumber != num+2:
			print "Error in the deployed sink and source."
			while len(self.G.nodeList) > num:
				self.G.nodeList.pop()
		else:
			for i in self.G.nodeList[self.sink].adj:
				self.G.nodeList[i].adj.add(self.sink)
			for i in self.G.nodeList[self.source].adj:
				self.G.nodeList[i].adj.add(self.source)

	def deployAttacker(self, Node):
		"""部署攻击者初始位置"""
		self.attacker.initDeploy(Node)

	def levelSourceSink(self):
		"构建网络层次: hops to source then hops to sink"
		firstHop = np.zeros(self.G.nodeNumber, dtype=np.int)
		# hops to source
		hop2Source = np.ones(self.G.nodeNumber, dtype=np.int) * (-1)
		queue = Queue.Queue()
		visited = np.zeros(self.G.nodeNumber, dtype=np.int8)
		src = self.source
		hop2Source[src] = 0
		queue.put(src)
		visited[src] = 1
		while not queue.empty():
			u = queue.get()
			visited[u] = 0
			for v in self.G.nodeList[u].adj:
				if hop2Source[v] == -1 or hop2Source[u] + 1 < hop2Source[v]:
					hop2Source[v] = hop2Source[u] + 1
					# update firstHop
					firstHop[v] = hop2Source[v] if firstHop[v] == 0 else firstHop[v]
					if visited[v] == 0:
						queue.put(v)
						visited[v] = 1
		# hops to sink
		hop2Sink = np.ones(self.G.nodeNumber, dtype=np.int) * (-1)
		queue = Queue.Queue()
		visited = np.zeros(self.G.nodeNumber, dtype=np.int8)
		src = self.sink
		hop2Sink[src] = 0
		queue.put(src)
		visited[src] = 1
		while not queue.empty():
			u = queue.get()
			visited[u] = 0
			for v in self.G.nodeList[u].adj:
				if hop2Sink[v] == -1 or hop2Sink[u] + 1 < hop2Sink[v]:
					hop2Sink[v] = hop2Sink[u] + 1
					if visited[v] == 0:
						queue.put(v)
						visited[v] = 1
		# copy to cpsNode [hop2Source, hop2Sink, firstHop]
		for node in self.G.nodeList:
			self.G.nodeList[node.identity].level = [hop2Source[node.identity], hop2Sink[node.identity],
													firstHop[node.identity]]
		return max(firstHop)

	def fakeSourceSelection(self, Ti, firstHopMAX):
		"""
		temporary fake sources 和 permanent fake sources flooding fake messages
		:param Ti: 周期数
		:return: self.G.nodeList.state, self.G.nodeList.weight => 'TFS'/'PFS', (Duration, Period)
		"""
		if Ti == 1:
			FS = [node.identity for node in self.G.nodeList if node.level[1] == 1]
			for v in FS:
				self.G.nodeList[v].state = 'TFS'
				self.G.nodeList[v].weight = (0.5, 2)
		else:
			preFS = [node.identity for node in self.G.nodeList if node.state == 'TFS']
			for u in preFS:
				for v in self.G.nodeList[u].adj:
					if v not in preFS and self.G.nodeList[v].state == 'NORMAL' \
							and self.G.nodeList[v].level[0] > 0.8*self.G.nodeList[self.sink].level[0]:
						if self.G.nodeList[v].level[2] >= firstHopMAX-1:
							self.G.nodeList[v].state = 'PFS'
							self.G.nodeList[v].weight = (self.Tmax, 1)
						else:
							self.G.nodeList[v].state = 'TFS'
							self.G.nodeList[v].weight = (1.0, min(2*self.G.nodeList[v].level[1], 4))
			for u in preFS:
				self.G.nodeList[u].state = 'NORMAL'

	def updateAdjMatrix(self):
		"""
		更新通信关系矩阵，即建立 'TFS'/'PFS' 与其他节点间的通信关系
		:param self.G.adjacentMatrix: 通信关系矩阵
		:return: self.G.adjacentMatrix
		"""
		self.G.initAdjMatrix()
		for node in self.G.nodeList:
			if node.state == 'TFS' or node.state == 'PFS':
				u = node.identity
				queue = Queue.Queue()
				visited = np.zeros(self.G.nodeNumber, dtype=np.int8)
				queue.put(u)
				visited[u] = 1
				while not queue.empty():
					u = queue.get()
					for v in self.G.nodeList[u].adj:
						if visited[v] == 0:
							self.G.adjacentMatrix[u,v] += node.weight[1]
							queue.put(v)
							visited[v] = 1

	def sendSource2Sink(self, Ti):
		""" TO DO SOMETHING
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
        :param Ti: 周期数
		:return: flag, delayTi, energyTi(MAX) 
		"""
		flag = False
		packetSize = 500 * 8  # 单次数据包大小 bit
		# 路由时延
		delays  = np.ones(self.G.nodeNumber, dtype=np.float)*np.inf
		visited = np.zeros(self.G.nodeNumber, dtype=np.int8)
		queue   = Queue.Queue()
		queue.put(self.source)
		delays[self.source]  = 0
		visited[self.source] = 1
		while not queue.empty():
			u = queue.get()
			for v in self.G.nodeList[u].adj:
				d = self.G.delayModelusingAdjMatrix(u, v)
				if delays[v] == np.inf or delays[v] > delays[u] + d:
					if delays[v] == np.inf:		# source floods to sink 产生的通信链路
						self.G.adjacentMatrix[u,v] += 1
					delays[v] = delays[u] + d
					if visited[v] == 0:
						queue.put(v)
						visited[v] = 1
		delayTi = delays[self.sink] * packetSize
		# 网络能耗 self.G.energyModelusingAdjMatrix()
		energyTi = 0
		for Node in self.G.nodeList:
			ec = self.G.energyModelusingAdjMatrix(Node)
			ec *= packetSize
			energyTi += ec
			self.G.nodeList[Node.identity].energy -= ec
			if not self.G.nodeList[Node.identity].isAlive():
				flag = True
		self.attacker.move(self.G)
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
		each_result.append(self.G.nodeList[self.sink].level[0])
		# number of transmitted and broadcast fake messages
		each_result.append(np.max(self.G.adjacentMatrix, axis=1).sum() - self.G.nodeNumber + 1)
		self.result.append(each_result)

	def algDynamicFSS(self):
		"""
		动态虚假源选择算法的主函数
		1）虚假源节点选择，包含 TFS 和 PFS
		2）更新当前轮的通信矩阵
		3）源节点向基站发送消息
		:return: 
		"""
		self.generateSINKandSOURCE()
		self.deployAttacker(self.G.nodeList[self.sink])
		self.levelSourceSink()

		firstHopMAX = self.levelSourceSink()
		for Ti in range(1, self.Tmax+1):
			# if Ti % 100 == 0:
			# 	print Ti
			# elif Ti % 10 == 0:
			# 	print Ti,
			self.fakeSourceSelection(Ti, firstHopMAX)
			self.updateAdjMatrix()
			flag, delayTi, energyTi = self.sendSource2Sink(Ti)
			self.listDelay.append(delayTi)
			self.listEnergyConsumption.append(energyTi)

			print '[%d] flag: %d, delayTi: %f, energyTi: %f' % (Ti, flag, delayTi/1e9, energyTi/self.G.nodeNumber)

			if flag or Ti == self.Tmax:
				self.safety = Ti
				break

	def resultPlot(self):
		print 'Safety is %d' % self.safety
		plt.figure(figsize=(12,8))
		plt.subplot(211)
		plt.plot(np.array(self.listDelay)/1e9)
		plt.legend(['delay'], loc=0)
		plt.subplot(212)
		plt.plot(np.array(self.listEnergyConsumption))
		plt.legend(['energy consumption'], loc=0)
		plt.show()

if __name__ == '__main__':
	network = cpsNetwork(file_path='load_network/network.csv')
	print '网络规模：', network.nodeNumber, network.areaLength
	dFFS = dynamicFSS(G=network, Tmax=2000, sink_pos=(-200,-200), source_pos=(200,200))
	# print ""
	# dFFS.generateSINKandSOURCE()
	# # dFFS.display()
	# # print ""
	# # dFFS.G.nodeList[dFFS.sink].display()
	# # print ""
	# # dFFS.G.nodeList[dFFS.source].display()
	#
	# dFFS.deployAttacker(dFFS.G.nodeList[dFFS.sink])
	# # dFFS.attacker.display()
	#
	# dFFS.levelSourceSink()
	# for node in dFFS.G.nodeList:
	# 	print node.identity, node.level, list(node.adj)
	#
	# dFFS.display()
	#
	dFFS.algDynamicFSS()
	print np.array(dFFS.result)
	# dFFS.display()
	# dFFS.G.display()
	dFFS.resultPlot()

	restEnergy = [dFFS.G.initEnergy - node.energy for node in dFFS.G.nodeList if
				  node.identity != dFFS.source and node.identity != dFFS.sink]
	# print restEnergy
	print max(restEnergy), np.mean(restEnergy), min(restEnergy), np.std(restEnergy)

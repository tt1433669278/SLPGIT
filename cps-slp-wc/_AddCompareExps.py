# coding=utf-8
"""
新增的对比实验，内容如下：
1）(位置隐私水平)攻击者与 source 的跳数，min hop between attacker and source among Ti in Tsafe
2）(网络传输时延)max hop between attacker and source among Ti in Tsafe
3）(网络能耗)   每周期虚假消息广播及中转次数 1/|Tsafe|*sum(The number of broadcast and transmitted fake message among Ti in Tsafe)
	
	def sendSource2Sink(self):
		self.getResult4AnalysisEachRound(Ti)
	
	def getResult4AnalysisEachRound(self, Ti):

"""

from FakeSourceScheduling import *
from PhantomRouting import *
from TreeBranch import *
from dynamicFSS import *

def getHopAttacker2Source(G, attacker):
	"""
	计算攻击者到过的距离源节点最近的位置
	:param G: 
	:param attacker: 
	:return: min hop between attacker and source
	"""
	retHop = np.inf

	for src in attacker.trace:
		if G.nodeList[src].state == 'SOURCE':
			retHop = 0
			break

		queue   = Queue.Queue()
		hops    = np.ones(G.nodeNumber) * (-1)
		visited = np.zeros(G.nodeNumber)

		queue.put(src)
		hops[src]    = 0
		visited[src] = 1

		while not queue.empty():
			u = queue.get()
			visited[u] = 0
			for v in G.nodeList[u].adj:
				if hops[v] == -1 or hops[v] > hops[u] + 1:
					hops[v] = hops[u] + 1
					if hops[v] < retHop and visited[v] == 0:
						queue.put(v)
						visited[v] = 1
					if G.nodeList[v].state == 'SOURCE':
						retHop = min(retHop, hops[v])
	return retHop

def getHopSource2Sink(result):
	"""
	计算基站与源节点的最大跳数
	计算每个周期的虚假消息转发次数打平均值
	:param result: [hop from sink to source, number of fake messages] 
	:return: 
	"""
	result = np.array(result)
	return result[:,0].max()

def getNumberFakeMessages(result):
	"""
	计算基站与源节点的最大跳数
	计算每个周期的虚假消息转发次数打平均值
	:param result: [hop from sink to source, number of fake messages] 
	:return: 
	"""
	result = np.array(result)
	return result[:,1].mean()

nodeNumber, areaLength, initEnergy, radius = 1000, 1000, 1e9, 50
file_names = ['load_network/%d-Network_%d_%d_%d_%d.csv' % (x, nodeNumber, areaLength, initEnergy, radius)
			  	for x in range(1,51,1)]
sink_pos   = (-0.33 * areaLength, 0)
source_pos = (0.33 * areaLength, 0)

# phantom routing
print '================================ Phantom Routing ================================'
csvfile = file('Exps/phantom_routing_add.csv', 'w')
writer = csv.writer(csvfile)
HopsAttacker, HopsSourceSink, NumberFakes = [], [], []
for file_name in file_names:
	print file_name
	network = cpsNetwork(file_path=file_name)
	alg = routingPhantom(G=network,
						Tmax=2000, Hwalk=10,
						sink_pos=sink_pos, source_pos=source_pos)
	alg.phantomRouting()
	HopsAttacker.append(getHopAttacker2Source(alg.G, alg.attacker))
	HopsSourceSink.append(getHopSource2Sink(alg.result))
	NumberFakes.append(getNumberFakeMessages(alg.result))

writer.writerow(['HopsAttacker', 'HopsSourceSink', 'NumberFakes'])
for p1, p2, p3 in zip(HopsAttacker, HopsSourceSink, NumberFakes):
	writer.writerow([p1, p2, p3])
csvfile.close()

# # treebranch
# print '\n================================ Treebranch ================================'
# csvfile = file('Exps/treebranch_add.csv', 'w')
# writer = csv.writer(csvfile)
# HopsAttacker, HopsSourceSink, NumberFakes = [], [], []
# for file_name in file_names:
# 	print file_name
# 	network = cpsNetwork(file_path=file_name)
# 	alg = routingTreeBranch(G=network,
# 							Tmax=2000, Hwalk=10, Theta=0.2, PHI=5, C_Branch=0.65, gap=10,
# 							sink_pos=sink_pos, source_pos=source_pos)
# 	alg.treebranchRouting()
# 	HopsAttacker.append(getHopAttacker2Source(alg.G, alg.attacker))
# 	HopsSourceSink.append(getHopSource2Sink(alg.result))
# 	NumberFakes.append(getNumberFakeMessages(alg.result))
#
# writer.writerow(['HopsAttacker', 'HopsSourceSink', 'NumberFakes'])
# for p1, p2, p3 in zip(HopsAttacker, HopsSourceSink, NumberFakes):
# 	writer.writerow([p1, p2, p3])
# csvfile.close()
#
# # dynamicFSS
# print '\n================================ DynamicFSS ================================'
# csvfile = file('Exps/dynamicFSS_add.csv', 'w')
# writer = csv.writer(csvfile)
# HopsAttacker, HopsSourceSink, NumberFakes = [], [], []
# for file_name in file_names:
# 	print file_name
# 	network = cpsNetwork(file_path=file_name)
# 	alg = dynamicFSS(G=network,
# 					 Tmax=2000,
# 					 sink_pos=sink_pos, source_pos=source_pos)
# 	alg.algDynamicFSS()
# 	HopsAttacker.append(getHopAttacker2Source(alg.G, alg.attacker))
# 	HopsSourceSink.append(getHopSource2Sink(alg.result))
# 	NumberFakes.append(getNumberFakeMessages(alg.result))
#
# writer.writerow(['HopsAttacker', 'HopsSourceSink', 'NumberFakes'])
# for p1, p2, p3 in zip(HopsAttacker, HopsSourceSink, NumberFakes):
# 	writer.writerow([p1, p2, p3])
# csvfile.close()
#
# # cpstopoFakeSourceScheduling
# print '\n================================ cpstopoFakeSourceScheduling ================================'
# c_capture, c_alpha, c_beta = 1e-40, 0.02, 0.6
# csvfile = file('Exps/cpstopoFakeSourceScheduling_add.csv', 'w')
# writer = csv.writer(csvfile)
# HopsAttacker, HopsSourceSink, NumberFakes = [], [], []
# for file_name in file_names:
# 	print file_name
# 	network = cpsNetwork(file_path=file_name)
# 	alg = cpstopoFakeScheduling(G=network,
# 								Tmax=2000, c_capture=c_capture, c_alpha=c_alpha, c_beta=c_beta,
# 								sink_pos=sink_pos, source_pos=source_pos)
# 	alg.fakeScheduling()
# 	HopsAttacker.append(getHopAttacker2Source(alg.G, alg.attacker))
# 	HopsSourceSink.append(getHopSource2Sink(alg.result))
# 	NumberFakes.append(getNumberFakeMessages(alg.result))
#
# writer.writerow(['HopsAttacker', 'HopsSourceSink', 'NumberFakes'])
# for p1, p2, p3 in zip(HopsAttacker, HopsSourceSink, NumberFakes):
# 	writer.writerow([p1, p2, p3])
# csvfile.close()
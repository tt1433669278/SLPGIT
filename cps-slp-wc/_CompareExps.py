# coding=utf-8
"""
算法灵敏度分析：
1）alpha, beta 联合分析
2）被捕或概率分析
性能指标同对比算法的性能指标

对比实验，对比算法如下：
1）自己提出的算法
2）幻影路由技术
3）基于树的诱导路由技术
4）动态虚假源算法
算法性能对比指标：
1）安全周期数 Tsafe，指网络所有节点都存活，且源节点未被捕获的周期数
2）传输时延 TR，指 source 向 sink 发送数据所花费的时间，取安全周期下的最大值
3）网络能耗 Ei，指安全周期内能耗最多的节点每周期所花费的平均能量
"""
from FakeSourceScheduling import *
from PhantomRouting import *
from TreeBranch import *
from dynamicFSS import *

def getEi(G, initEnergy):
	Ei = 0
	for node in G.nodeList:
		if node.state != 'SINK' and node.state != 'SOURCE':
			Ei = max(Ei, initEnergy-node.energy)
	return Ei

nodeNumber, areaLength, initEnergy, radius = 1000, 1000, 1e9, 50
file_names = ['load_network/%d-Network_%d_%d_%d_%d.csv' % (x, nodeNumber, areaLength, initEnergy, radius)
			  	for x in range(1,51,1)]
sink_pos   = (-0.33 * areaLength, 0)
source_pos = (0.33 * areaLength, 0)

# phantom routing
print ''
print 'phantom routing'
csvfile = file('Exps/phantom_routing.csv', 'w')
writer = csv.writer(csvfile)
Tsafe = list()
TR = list()
Ei = list()
for file_name in file_names:
	print file_name
	for i in range(1):
		network = cpsNetwork(file_path=file_name)
		alg = routingPhantom(G=network,
							Tmax=2000, Hwalk=10,
							sink_pos=sink_pos, source_pos=source_pos)
		alg.phantomRouting()
		Tsafe.append(alg.safety)
		TR.append(np.max(alg.listDelay))
		Ei.append(getEi(alg.G, initEnergy) * 1.0 / alg.safety)
		print 'Debug:', alg.G.nodeList[alg.sink].level, alg.G.nodeList[alg.source].level

Ts_list = ['Tsafe']
TR_list = ['TR']
Ei_list = ['Ei']

Ts_list.extend(Tsafe)
TR_list.extend(TR)
Ei_list.extend(Ei)

writer.writerow(Ts_list)
writer.writerow(TR_list)
writer.writerow(Ei_list)

csvfile.close()

# treebranch
print ''
print 'treebranch'
csvfile = file('Exps/treebranch.csv', 'w')
writer = csv.writer(csvfile)
Tsafe = list()
TR = list()
Ei = list()
for file_name in file_names:
	print file_name
	for i in range(1):
		network = cpsNetwork(file_path=file_name)
		alg = routingTreeBranch(G=network,
								Tmax=2000, Hwalk=10, Theta=0.2, PHI=5, C_Branch=0.65, gap=10,
								sink_pos=sink_pos, source_pos=source_pos)
		alg.treebranchRouting()
		Tsafe.append(alg.safety)
		TR.append(np.max(alg.listDelay))
		Ei.append(getEi(alg.G, initEnergy) * 1.0 / alg.safety)
		print 'Debug:', alg.G.nodeList[alg.sink].level, alg.G.nodeList[alg.source].level

Ts_list = ['Tsafe']
TR_list = ['TR']
Ei_list = ['Ei']

Ts_list.extend(Tsafe)
TR_list.extend(TR)
Ei_list.extend(Ei)

writer.writerow(Ts_list)
writer.writerow(TR_list)
writer.writerow(Ei_list)

csvfile.close()

# dynamicFSS
print ''
print 'dynamicFSS'
csvfile = file('Exps/dynamicFSS.csv', 'w')
writer = csv.writer(csvfile)
Tsafe = list()
TR = list()
Ei = list()
for file_name in file_names:
	print file_name
	for i in range(1):
		network = cpsNetwork(file_path=file_name)
		alg = dynamicFSS(G=network,
						Tmax=2000,
						sink_pos=sink_pos, source_pos=source_pos)
		alg.algDynamicFSS()
		Tsafe.append(alg.safety)
		TR.append(np.max(alg.listDelay))
		Ei.append(getEi(alg.G, initEnergy) * 1.0 / alg.safety)
		print 'Debug:', alg.G.nodeList[alg.sink].level, alg.G.nodeList[alg.source].level

Ts_list = ['Tsafe']
TR_list = ['TR']
Ei_list = ['Ei']

Ts_list.extend(Tsafe)
TR_list.extend(TR)
Ei_list.extend(Ei)

writer.writerow(Ts_list)
writer.writerow(TR_list)
writer.writerow(Ei_list)

csvfile.close()

# cpstopoFakeSourceScheduling
print ''
print 'cpstopoFakeSourceScheduling'
c_capture, c_alpha, c_beta = 1e-40, 0.02, 0.6
csvfile = file('Exps/cpstopoFakeSourceScheduling.csv', 'w')
writer = csv.writer(csvfile)
Tsafe = list()
TR = list()
Ei = list()
for file_name in file_names:
	print file_name
	for i in range(1):
		network = cpsNetwork(file_path=file_name)
		alg = cpstopoFakeScheduling(G=network,
						Tmax=2000, c_capture=c_capture, c_alpha=c_alpha, c_beta=c_beta,
						sink_pos=sink_pos, source_pos=source_pos)
		alg.fakeScheduling()
		Tsafe.append(alg.safety)
		TR.append(np.max(alg.listDelay))
		Ei.append(getEi(alg.G, initEnergy) * 1.0 / alg.safety)
		print 'Debug:', alg.G.nodeList[alg.sink].level, alg.G.nodeList[alg.source].level, \
			np.array([len(x) for x in alg.listFakeSource]).mean()
		print alg.attacker.trace
		print np.mean([len(lf) for lf in alg.listFakeSource])
		print ''

Ts_list = ['Tsafe']
TR_list = ['TR']
Ei_list = ['Ei']

Ts_list.extend(Tsafe)
TR_list.extend(TR)
Ei_list.extend(Ei)

writer.writerow(Ts_list)
writer.writerow(TR_list)
writer.writerow(Ei_list)

csvfile.close()

# again
print 'cpstopoFakeSourceScheduling ... again'
csvfile = file('Exps/cpstopoFakeSourceScheduling_again.csv', 'w')
writer = csv.writer(csvfile)
Tsafe = list()
TR = list()
Ei = list()
for file_name in file_names:
	print file_name
	for i in range(1):
		network = cpsNetwork(file_path=file_name)
		alg = cpstopoFakeScheduling(G=network,
						Tmax=2000, c_capture=c_capture, c_alpha=c_alpha, c_beta=c_beta,
						sink_pos=sink_pos, source_pos=source_pos)
		alg.fakeScheduling()
		Tsafe.append(alg.safety)
		TR.append(np.max(alg.listDelay))
		Ei.append(getEi(alg.G, initEnergy) * 1.0 / alg.safety)
		print 'Debug:', alg.G.nodeList[alg.sink].level, alg.G.nodeList[alg.source].level, \
			np.array([len(x) for x in alg.listFakeSource]).mean()
		print alg.attacker.trace
		print np.mean([len(lf) for lf in alg.listFakeSource])
		print ''

Ts_list = ['Tsafe']
TR_list = ['TR']
Ei_list = ['Ei']

Ts_list.extend(Tsafe)
TR_list.extend(TR)
Ei_list.extend(Ei)

writer.writerow(Ts_list)
writer.writerow(TR_list)
writer.writerow(Ei_list)

csvfile.close()
# coding=utf-8
"""
算法灵敏度分析：
1）alpha, beta 联合分析
2）被捕或概率分析
性能指标同对比算法的性能指标
1）安全周期数 Tsafe，指网络所有节点都存活，且源节点未被捕获的周期数
2）传输时延 TR，指 source 向 sink 发送数据所花费的时间，取安全周期下的最大值
3）网络能耗 Ei，指安全周期内能耗最多的节点每周期所花费的平均能量
"""
from FakeSourceScheduling import *
from cpsNetwork import *

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

# capture
print ''
print 'capture: 1e-0: 1e-2 : 1e-64'
csvfile = file('Exps/sensitivity_analysis_capture.csv', 'w')
writer = csv.writer(csvfile)
rng = list(range(-0, -65, -2))
for i, x in enumerate(rng):
	rng[i] = 10**x
for capture in rng:
	print 'capture:', capture
	Tsafe = list()
	TR = list()
	Ei = list()
	for file_name in file_names:
		print file_name
		for i in range(1):
			network = cpsNetwork(file_path=file_name)
			alg = cpstopoFakeScheduling(G=network,
										Tmax=2000, c_capture=capture, c_alpha=0.02, c_beta=0.8,
										sink_pos=sink_pos, source_pos=source_pos)
			alg.fakeScheduling()
			Tsafe.append(alg.safety)
			TR.append(np.max(alg.listDelay))
			Ei.append(getEi(alg.G, initEnergy) * 1.0 / alg.safety)
			print 'Debug:', alg.G.nodeList[alg.sink].level, alg.G.nodeList[alg.source].level, \
				np.array([len(x) for x in alg.listFakeSource]).mean()
			print alg.attacker.trace
			print ''
	rs = [0.02, 0.8, capture]

	Ts_list = ['Tsafe']
	TR_list = ['TR']
	Ei_list = ['Ei']

	Ts_list.extend(rs)
	Ts_list.extend(Tsafe)

	TR_list.extend(rs)
	TR_list.extend(TR)

	Ei_list.extend(rs)
	Ei_list.extend(Ei)

	writer.writerow(Ts_list)
	writer.writerow(TR_list)
	writer.writerow(Ei_list)
csvfile.close()

# alpha
print ''
print 'alpha: 0.00: 0.01 : 0.20'
csvfile = file('Exps/sensitivity_analysis_alpha.csv', 'w')
writer = csv.writer(csvfile)
rng = list(range(0, 21, 1))
for i, x in enumerate(rng):
	rng[i] /= 100.0
for alpha in rng:
	print 'alpha:', alpha
	Tsafe = list()
	TR = list()
	Ei = list()
	for file_name in file_names:
		print file_name
		for i in range(1):
			network = cpsNetwork(file_path=file_name)
			alg = cpstopoFakeScheduling(G=network,
										Tmax=2000, c_capture=1e-32, c_alpha=alpha, c_beta=0.8,
										sink_pos=sink_pos, source_pos=source_pos)
			alg.fakeScheduling()
			Tsafe.append(alg.safety)
			TR.append(np.max(alg.listDelay))
			Ei.append(getEi(alg.G, initEnergy) * 1.0 / alg.safety)
			print 'Debug:', alg.G.nodeList[alg.sink].level, alg.G.nodeList[alg.source].level, \
				np.array([len(x) for x in alg.listFakeSource]).mean()
			print alg.attacker.trace
			print ''
	rs = [alpha, 0.8, 1e-32]

	Ts_list = ['Tsafe']
	TR_list = ['TR']
	Ei_list = ['Ei']

	Ts_list.extend(rs)
	Ts_list.extend(Tsafe)

	TR_list.extend(rs)
	TR_list.extend(TR)

	Ei_list.extend(rs)
	Ei_list.extend(Ei)

	writer.writerow(Ts_list)
	writer.writerow(TR_list)
	writer.writerow(Ei_list)
csvfile.close()

# beta
print ''
print 'beta: 0.0: 0.1 : 1.0'
csvfile = file('Exps/sensitivity_analysis_beta.csv', 'w')
writer = csv.writer(csvfile)
rng = list(range(0, 11, 1))
for i, x in enumerate(rng):
	rng[i] /= 10.0
for beta in rng:
	print 'beta:', beta
	Tsafe = list()
	TR = list()
	Ei = list()
	for file_name in file_names:
		print file_name
		for i in range(1):
			network = cpsNetwork(file_path=file_name)
			alg = cpstopoFakeScheduling(G=network,
										Tmax=2000, c_capture=1e-32, c_alpha=0.02, c_beta=beta,
										sink_pos=sink_pos, source_pos=source_pos)
			alg.fakeScheduling()
			Tsafe.append(alg.safety)
			TR.append(np.max(alg.listDelay))
			Ei.append(getEi(alg.G, initEnergy) * 1.0 / alg.safety)
			print 'Debug:', alg.G.nodeList[alg.sink].level, alg.G.nodeList[alg.source].level, \
				np.array([len(x) for x in alg.listFakeSource]).mean()
			print alg.attacker.trace
			print ''
	rs = [0.02, beta, 1e-32]

	Ts_list = ['Tsafe']
	TR_list = ['TR']
	Ei_list = ['Ei']

	Ts_list.extend(rs)
	Ts_list.extend(Tsafe)

	TR_list.extend(rs)
	TR_list.extend(TR)

	Ei_list.extend(rs)
	Ei_list.extend(Ei)

	writer.writerow(Ts_list)
	writer.writerow(TR_list)
	writer.writerow(Ei_list)
csvfile.close()

# # alpha, beta
# print ''
# print 'alpha-beta:'
# csvfile = file('Exps/sensitivity_analysis_alpha_beta.csv', 'w')
# writer = csv.writer(csvfile)
#
# rnga = list(range(1, 21))
# for i, x in enumerate(rnga):
# 	rnga[i] /= 100.0
# rngb = list(range(0, 101, 10))
# for i, x in enumerate(rngb):
# 	rngb[i] /=100.0
# for alpha in rnga:
# 	for beta in rngb:
# 		Tsafe = list()
# 		TR = list()
# 		Ei = list()
# 		for file_name in file_names:
# 			print file_name
# 			for i in range(1):
# 				network = cpsNetwork(file_path=file_name)
# 				alg = cpstopoFakeScheduling(G=network,
# 								   Tmax=2000, c_capture=1e-16, c_alpha=alpha, c_beta=beta,
# 								   sink_pos=(-0.33*areaLength, 0), source_pos=(0.33*areaLength, 0))
# 				alg.fakeScheduling()
# 				print alg.backbone
# 				print alg.listFakeSource[-1]
# 				Tsafe.append(alg.safety)
# 				TR.append(np.max(alg.listDelay))
# 				Ei.append(getEi(alg.G, initEnergy)*1.0/alg.safety)
# 		rs = [alpha, beta, '1e-16']
#
# 		Ts_list = ['Tsafe']
# 		TR_list = ['TR']
# 		Ei_list = ['Ei']
#
# 		Ts_list.extend(rs)
# 		Ts_list.extend(Tsafe)
#
# 		TR_list.extend(rs)
# 		TR_list.extend(TR)
#
# 		Ei_list.extend(rs)
# 		Ei_list.extend(Ei)
#
# 		writer.writerow(Ts_list)
# 		writer.writerow(TR_list)
# 		writer.writerow(Ei_list)
#
# csvfile.close()
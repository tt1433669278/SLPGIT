# coding=utf-8
"""
连通的网络生成，包含网络节点数量，正方形区域边长，节点初始能量，节点通信半径，而节点位置随机生成
生成的网络共 50 个，存储路径为 'load_network/*.csv'
生成的网络有：
1）nodeNumber, areaLength, initEnergy, radius = 1000, 1000, 1e9, 50，文件名以参数顺序命名

"""

from cpsNetwork import *

nodeNumber, areaLength, initEnergy, radius = 100, 1000, 1e9, 2  # 1000, 1000, 1e9, 50
file_names = ['load_network/%d-myNetwork_%d_%d_%d_%d.csv' % (x, nodeNumber, areaLength, initEnergy, radius)
			  	for x in range(1,2,1)]

for file_name in file_names:
	# generate connected network
	print ''
	print file_name
	flag = True
	gn = 1
	while flag:
		network = cpsNetwork(nodeNumber=nodeNumber, areaLength=areaLength, initEnergy=initEnergy, radius=radius)
		start = datetime.datetime.now()
		flag = network.generateNeighbors()
		end = datetime.datetime.now()
		print gn, ": It takes", end - start
		if not flag and network.isConnect():
			flag = False
		else:
			flag = True
		gn += 1
	network.display()
	if not flag:
		network.to_csv(file_name)
		network.display()
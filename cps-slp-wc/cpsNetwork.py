# coding=utf-8
from cpsNode import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import Queue


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

    def __init__(self, nodeNumber=50, areaLength=100, initEnergy=1e9, radius=50, file_path=''):
        """
		实例初始化
		:param nodeNumber 
		:param areaLength 
		:param initEnergy 
		:param radius
		:param file_path
		"""
        if len(file_path) != 0:
            csvfile = file(file_path, 'r')
            reader = csv.reader(csvfile)
            row_num = 0  # 记录当前读取的行数
            self.nodeList = []
            for line in reader:
                row = line
                if row_num == 0:
                    self.nodeNumber = np.int(row[0])
                    self.areaLength = np.float(row[1])
                    self.initEnergy = np.float(row[2])
                    self.radius = np.float(row[3])
                    row_num = 1
                else:
                    node = cpsNode(identity=np.int(row[0]), position=(np.float(row[1]), np.float(row[2])),
                                   energy=self.initEnergy, radius=self.radius, state='NORMAL',
                                   adj=set([np.int(x) for x in row[3:]]))
                    self.nodeList.append(node)
            self.adjacentMatrix = np.zeros((self.nodeNumber, self.nodeNumber),
                                           dtype=np.int)  # 创建一个全零的二维数组self.adjacentMatrix，self.nodeNumber，用于表示邻接矩阵
        else:
            self.nodeNumber = nodeNumber
            self.areaLength = areaLength
            self.initEnergy = initEnergy
            self.radius = radius
            self.nodeList = []  # [0, nodeNuber-1] are NORMAL nodes
            ###########################
            # # 仅复现用
            # SENSORS = [[x, y] for x in range(0, 10) for y in range(0, 10)]
            #
            # for identity, position in enumerate(SENSORS):
            #     node = cpsNode(identity=identity, position=position, energy=initEnergy, radius=radius, state='NORMAL',
            #                    adj=set())
            #     self.nodeList.append(node)
            ##############################
            # generate node position randomly
            for i in range(nodeNumber):
                position = (np.random.rand(2) - 0.5) * areaLength  # [-0.5*areaLength, 0.5*areaLength)
                position = (position[0], position[1])
                node = cpsNode(identity=i, position=position, energy=initEnergy, radius=radius, state='NORMAL',
                               adj=set())
                self.nodeList.append(node)

            self.adjacentMatrix = np.zeros((nodeNumber, nodeNumber), dtype=np.int)

    def initAdjMatrix(self):
        """邻接矩阵（通信关系）初始化"""
        self.adjacentMatrix = np.zeros((self.nodeNumber, self.nodeNumber), dtype=np.int)

    def addNode(self, Node):
        """增加节点"""
        # 增加邻居
        xy = np.array([[node.position[0], node.position[1]] for node in self.nodeList])
        idxy = np.array([node.identity for node in self.nodeList])
        x = Node.position[0]
        y = Node.position[1]
        ids = idxy[(xy[:, 0] > (x - self.radius)) & (xy[:, 0] < (x + self.radius))
                   & (xy[:, 1] > (y - self.radius)) & (xy[:, 1] < (y + self.radius))]
        for i in ids:
            if i != Node.identity and self.calculate2Distance(Node, self.nodeList[i]) < self.radius:
                Node.adj.add(i)
        if not len(Node.adj):
            return False
        self.nodeList.append(Node)
        self.nodeNumber += 1
        self.adjacentMatrix = np.zeros((self.nodeNumber, self.nodeNumber), dtype=np.int8)
        return True
# 输入两个点输出距离
    def calculate2Distance(self, node1, node2):
        """计算两个节点之间的欧式距离"""
        x1 = node1.position[0]
        y1 = node1.position[1]
        x2 = node2.position[0]
        y2 = node2.position[1]
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def calculate_angle(self, node1, node2):
        # 计算以源节点和汇聚节点连线为 x 轴的角度值
        dx = node1.position[0] - node2.position[0]
        dy = node1.position[1] - node2.position[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
# 输入两个点输出时延
    def delayModelusingAdjMatrix(self, u, v):
        """
		估计单个节点，单跳，单位 bit 数据的时延 用于模拟无线传感器网络中节点之间的单跳传输时延
		u -> v
		"""
        tuv = 100.  # 无干扰,bit,接收时间 ns
        pe = 0  # 环境的噪声功率
        guvl = 9.488e-5  # 无线设备参数，d < d0
        guvh = 5.0625  # 无线设备参数，d >= d0
        d0 = 231  # 跨空间距离阈值
        pt = 10e0  # 节点发送功率
        pv = 0  # 接收节点干扰功率
        puv = 0
        for neighbor in self.nodeList[v].adj:
            Neighbor = self.nodeList[neighbor]  # 邻居
            # 使用 .adjacentMatrix 判定两节点是否连通
            if self.adjacentMatrix[neighbor, v] != 0 and neighbor != u:
                d = self.calculate2Distance(Neighbor, self.nodeList[v])
                if d < d0:
                    puv = (guvl * pt) * (d * 2)
                else:
                    puv = (guvh * pt) * (d * 4)
                pv += puv
        puv = 0  # u->v 接收功率
        d = self.calculate2Distance(self.nodeList[u], self.nodeList[v])
        if d < d0:
            puv = (guvl * pt) / (d ** 2)
        else:
            puv = (guvh * pt) / (d ** 4)
        ruv = 0  # 信干噪比
        if pv > np.finfo(np.float64).eps:
            ruv = puv / (pe + pv)
        else:
            ruv = 20.
        # 范围单跳，单位 bit 时延
        return tuv / (1. - np.exp(ruv * (-0.5)))  #
# 节点  能耗
    def energyModelusingAdjMatrix(self, Node):
        """
		估计单个节点，单位 bit 数据，的能耗
		"""
        d0 = 231  # 跨空间距离阈值
        d = Node.radius  # 节点广播半径
        Eelec = 50.  # 单位 bit 数据能耗
        c_fs = 10e-3  # 自由空间下单位放大器能耗
        c_mp = 0.0013e-3  # 存在多径损耗打单位放大器能
        # 接收能耗
        rE = 0  # 单位数据接收能耗
        rE_num = 0
        tE_num = 0
        flag = False  # broadcast 标志位
        for neighbor in Node.adj:
            # neighbor send message to node.identity
            if self.adjacentMatrix[neighbor, Node.identity] > 0:  # 接收
                rE += Eelec
                rE_num += self.adjacentMatrix[neighbor, Node.identity]
            # 广播消息标志
            if self.adjacentMatrix[Node.identity, neighbor] > 0:  # 发送
                flag = True
                tE_num += self.adjacentMatrix[Node.identity, neighbor]
        # 发送能耗
        tE = 0  # 单位数据发送能耗
        if flag:
            tE += Eelec
            if d < d0:
                tE += c_fs * (d ** 2)
            else:
                tE += c_mp * (d ** 4)
        return rE * rE_num + tE * tE_num

    def generateNeighbors(self):
        """建立邻居列表"""
        flag = False
        xy = np.array([[node.position[0], node.position[1]] for node in self.nodeList])
        idxy = np.array([node.identity for node in self.nodeList])
        for node in self.nodeList:
            x = node.position[0]
            y = node.position[1]
            ids = idxy[(xy[:, 0] > (x - self.radius)) & (xy[:, 0] < (x + self.radius))
                       & (xy[:, 1] > (y - self.radius)) & (xy[:, 1] < (y + self.radius))]
            for i in ids:
                if i != node.identity and self.calculate2Distance(node, self.nodeList[i]) < self.radius:
                    node.adj.add(i)
            if len(node.adj) == 0:
                flag = True
        return flag

    def isConnect(self):
        """网络连通性检测,记录节点是否已经被访问过 广度优先"""
        num = 0
        queue = Queue.Queue()
        visited = np.zeros(self.nodeNumber, dtype=np.int8)  # 记录节点是否已经被访问过
        src = 0
        queue.put(src)
        visited[src] = 1
        num += 1
        while not queue.empty():
            u = queue.get()
            for v in self.nodeList[u].adj:
                if visited[v] == 0:
                    queue.put(v)
                    visited[v] = 1
                    num += 1
        if num == self.nodeNumber:
            return True
        return False

    def to_csv(self, file_path):
        """
		存储网络，前提，已经建立邻居列表
		:param file_path: 
		:return: 
		"""
        csvfile = file(file_path, 'w')
        writer = csv.writer(csvfile)
        writer.writerow([self.nodeNumber, self.areaLength, self.initEnergy, self.radius])
        for node in self.nodeList:
            row = [node.identity, node.position[0], node.position[1]] + list(node.adj)
            writer.writerow(row)
        csvfile.close()

    def read_csv(self, file_path):
        """
		读取网络
		:param file_path: 
		:return: 
		"""
        csvfile = file(file_path, 'r')
        reader = csv.reader(csvfile)
        row_num = 0
        self.nodeList = []
        for line in reader:
            row = line
            if row_num == 0:
                self.nodeNumber = np.int(row[0])
                self.areaLength = np.float(row[1])
                self.initEnergy = np.float(row[2])
                self.radius = np.float(row[3])
                row_num = 1
            else:
                node = cpsNode(identity=np.int(row[0]), position=(np.float(row[1]), np.float(row[2])),
                               energy=self.initEnergy, radius=self.radius, state='NORMAL',
                               adj=set([np.int(x) for x in row[3:]]))
                self.nodeList.append(node)

    def display(self):
        print "节点总数：", self.nodeNumber
        print "正方形区域边长：", self.areaLength
        print "节点初始能量：", self.initEnergy
        print "节点半径：", self.radius
        # "点图"
        # temp_x = []
        # temp_y = []
        # for i in range(self.nodeNumber):
        #     # self.nodeList[i].display()
        #     temp_x.append(self.nodeList[i].position[0])
        #     temp_y.append(self.nodeList[i].position[1])
        # # print self.adjacentMatrix
        # plt.figure(figsize=(5, 5))
        # plt.plot(temp_x, temp_y, linewidth=0, color='gold', marker='p')
        # plt.axis("equal")
        # plt.show()


from FakeSourceScheduling import *
from MYFSS import *

if __name__ == '__main__':
    file_path = 'load_network/network.csv'  # csv file path

    # generate connected network
    flag = True
    gn = 30
    while flag and gn:
        network = cpsNetwork(nodeNumber=1000, radius=50, areaLength=500, initEnergy=1e9)
        start = datetime.datetime.now()
        flag = network.generateNeighbors()
        end = datetime.datetime.now()
        print gn, ": It takes", end - start
        if not flag and network.isConnect():
            flag = False
        else:
            flag = True
        gn -= 1
    print 'flag: %d' % flag
    network.display()
    if not flag:
        network.to_csv(file_path)
        network.display()
        for node in network.nodeList:
            print node.identity, list(node.adj)

    # read network from csv file
    if not flag:
        network = cpsNetwork(file_path=file_path)
        adj_len = []
        for node in network.nodeList:
            adj_len.append(len(node.adj))
            print node.identity, list(node.adj)
        print np.max(adj_len), np.mean(adj_len), np.min(adj_len), np.std(adj_len)
        # test on FakeSourceScheduling
        print '网络规模：', network.nodeNumber, network.areaLength

        fs = cpstopoFakeScheduling(G=network,
                                   Tmax=100, c_capture=1e-32, c_alpha=0.0, c_beta=0.8,
                                   sink_pos=(-0.3 * network.areaLength, 0),  # -0.3*network.areaLength),
                                   source_pos=(0.3 * network.areaLength, 0))  # 0.3*network.areaLength))
        fs.fakeScheduling()
        print ''
        restEnergy = [fs.G.initEnergy - node.energy for node in fs.G.nodeList if
                      node.identity != fs.source and node.identity != fs.sink]
        print max(restEnergy), np.mean(restEnergy), min(restEnergy), np.std(restEnergy)
        print ''
        fs.attacker.display()
        fs.backbonePlot()
        fs.plotDelayandConsumption()
        # 每轮的虚假源节点数量
        a = [len(x) for x in fs.listFakeSource]
        print 'a', a
        plt.figure(figsize=(15, 3))
        plt.plot(a)
        plt.ylabel('The number of fake source')
        plt.show()

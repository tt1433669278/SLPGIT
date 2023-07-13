# coding=utf-8

class cpsNode:
    """
	Cyber-physical systems 节点：
	1）身份    2）位置    
	3）能量    4）通信半径  
	5）状态    6）权重
	7）level
	8）邻居列表
	9）被捕获似然

	state: SINK -> 基站
		   SOURCE -> 源节点
		   NORMAL -> 普通节点
		   BACKBONE -> 骨干节点
		   FAKE -> 虚假源

	--------------------
	变量成员：
	identity = 身份/编号
	position = 位置
	energy   = 能量
	radius   = 通信半径
	state    = 状态信息
	weight   = 权重信息，表示节点是否广播消息
	level    = 层次信息，用于描述与 sink 间打跳数
	adj      = 相邻节点集合
	--------------------
	方法成员：
	dispaly()
	"""

    def __init__(self, identity=-1, position=(0, 0), energy=1e9, radius=20, state=-1, weight=-1, adj=set()):
        self.identity = identity
        self.position = position
        self.energy = energy
        self.radius = radius
        self.state = state
        self.weight = weight  # 1:broadcast, 0:not broadcast
        self.level = -1  # 跳数
        self.adj = adj
        self.parent = None
        self.g_cost = 0

    def display(self):
        print "Node", self.identity, ":", self.position, self.energy, self.radius, self.state
        print "The weight is", self.weight
        print "The level is", self.level
        print "Its neighbors are", list(self.adj)

    def isAlive(self):
        # print self.identity, "is 00000"
        return True if self.energy > 1e7 else False


import numpy as np

if __name__ == '__main__':
    node = cpsNode(identity=100)
    # node.display()
    print 'SINK' == "SINK"
    print 1.0 * np.inf == np.inf
    print np.finfo(np.float64).eps

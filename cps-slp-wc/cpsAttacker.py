# coding=utf-8
from cpsNode import *
import numpy as np

class cpsAttacker:
	"""
	攻击者模型
	---------
	变量成员：
	position = 攻击者位置，cpsNode 类型
	trace    = 攻击者的历史轨迹信息，位置用节点编号表示
	---------
	方法成员：
	__init__(position)     = 初始化
	display()              =
	traceBack(backbone, G) = weight == 1 or backbone 作为可回溯的位置
	tracebackNetwork(G)    = weight == 1 作为可回溯的位置
	"""

	def __init__(self, position=cpsNode()):
		self.position = position
		self.trace = []

	def initDeploy(self, Node):
		self.position = Node
		self.trace = [Node.identity]

	def display(self):
		# self.position.display()
		print 'The trace is', self.trace
		self.trace = []

	def move(self, G):
		"根据当前时间段所监听到的消息及相应的传输通道，随机选择下一跳回溯"
		bestNode = -1
		bestLikelihood = -1
		u = self.position.identity
		for v in self.position.adj:
			if G.adjacentMatrix[v,u] != 0:
				RAND = np.random.rand()
				if RAND > bestLikelihood:
					bestLikelihood = RAND
					bestNode = v
		if bestLikelihood != -1:
			self.trace.append(bestNode)
			self.position = G.nodeList[bestNode]
		else:
			self.trace.append(self.position.identity)

if __name__ == '__main__':
	position = cpsNode(identity=0)
	attacker = cpsAttacker(position=position)
	attacker.display()
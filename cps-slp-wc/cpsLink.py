# coding=utf-8
from cpsNode import *


class cpsLink:
	"""
    Cyber-physical systems 链路
        1）nodeFrom    2）nodeTo
        3）权重
    """

	def __init__(self, nodeFrom=cpsNode(), nodeTo=cpsNode(), weight=-1):
		self.nodeFrom = nodeFrom
		self.nodeTo = nodeTo
		self.weight = weight

	def display(self):
		print 'Link:', self.nodeFrom.identity, self.nodeTo.identity, self.weight


if __name__ == '__main__':
	fromNode = cpsNode(identity=0)
	toNode = cpsNode(identity=1)
	link = cpsLink(fromNode, toNode, 10)
	link.display()

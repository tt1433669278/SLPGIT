# -*- coding: utf-8 -*-
import heapq

from MYFSS import *
from cpsNetwork import *


class AStar:
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
        self.open_set = []
        self.closed_set = []

    def calculate_distance(self, node1, node2):
        a = ((node1.position[0] - node2.position[0]) ** 2 + (
                    node1.position[1] - node2.position[1]) ** 2) ** 0.5  # 计算两个节点之间的距离（例如欧氏距离）
        return a

    def heuristic(self, node, target):
        b = self.calculate_distance(node, target)  # 启发式函数，估计从当前节点到目标节点的距离（例如欧氏距离）
        return b

    def reconstruct_path(self, node):  # 从目标节点回溯到起始节点，构建最短路径
        path = []
        current = node
        while current is not None:
            path.append(current.identity)
            current = current.parent
        path.reverse()
        return path

    def find_shortest_path(self, start, target):
        start.g_cost = 0
        start.h_cost = self.heuristic(start, target)
        start.f_cost = start.g_cost + start.h_cost
        heapq.heappush(self.open_set, start)

        while self.open_set:
            current = heapq.heappop(self.open_set)
            if current == target:
                # 找到最短路径
                return self.reconstruct_path(current)

            self.closed_set.append(current)

            for neighbor in current.neighbors:
                if neighbor in self.closed_set:
                    continue

                g_temp = current.g_cost + self.calculate_distance(current, neighbor)

                if g_temp < neighbor.g_cost:
                    neighbor.parent = current
                    neighbor.g_cost = g_temp
                    neighbor.h_cost = self.heuristic(neighbor, target)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

                    if neighbor not in self.open_set:
                        heapq.heappush(self.open_set, neighbor)

        # 未找到路径
        return None


if __name__ == '__main__':
    network = network = cpsNetwork(file_path='load_network/network.csv')
    astar = AStar()
    fs = cpstopoFakeScheduling(G=network,
                               Tmax=100, c_capture=1e-80, c_alpha=0.05, c_beta=0.2,
                               sink_pos=(-200, -200), source_pos=(200, 200))
    # 设置起始节点和目标节点
    start_node = network.nodeList[3001]
    target_node = network.nodeList[3000]

    # 寻找最短路径
    shortest_path = astar.find_shortest_path(start_node, target_node)

    if shortest_path is not None:
        print("最短路径：", shortest_path)
    else:
        print("未找到路径")

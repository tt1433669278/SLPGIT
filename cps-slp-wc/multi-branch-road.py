# coding=utf-8
"""
# 对比算法
* phantom routing
* 基于树结构的诱导路由技术
* 动态虚假源选择算法
*多分支路径
## 多分支路径
Energy Balanced Source Location Privacy Scheme Using Multibranch Path in WSNs for IoT
Wireless Communications & Mobile Computing

"""
import math
import random

from cpsNetwork import *
from cpsAttacker import *
from cpsNode import *
import datetime


class multiroad:
    """
	要求：Hwalk + PHI < SOURCE.level
	"""

    def __init__(self, G=cpsNetwork(nodeNumber=10, areaLength=20, initEnergy=1e6, radius=50),
                 Tmax=1000, Hwalk=30, Theta=np.random.rand(), PHI=10, C_Branch=0.65, gap=10,
                 sink_pos=(0, 0), source_pos=(9, 9)):
        self.branchi_path = []
        self.MHR_path = []
        path = []
        self.num_branch_path = []
        self.RW_set = []
        self.Ni_Next_hop = None
        self.G = G

        self.Tmax = Tmax
        self.Hwalk = Hwalk
        self.Theta = Theta
        self.PHI = PHI
        self.C_Branch = C_Branch
        self.gap = gap

        self.sink = -1
        self.sink_pos = sink_pos
        self.source = -1
        self.source_pos = source_pos

        self.attacker = cpsAttacker()

        self.safety = -1
        self.listDelay = []
        self.listEnergyConsumption = []

        self.path = []
        self.bron0_path = []
        self.broni_path = []
        self.sum_path = []

    def display(self):
        print "节点总数 ", self.G.nodeNumber
        print "区域边长 ", self.G.areaLength
        print "节点初始能量 ", self.G.initEnergy
        print "节点最大通信半径 ", self.G.radius
        print "最大周期数 ", self.Tmax
        print "随机步数 ", self.Hwalk
        print "theta ", self.Theta
        print "PHI ", self.PHI
        print "sink ", self.sink, self.sink_pos
        print "source ", self.source, self.source_pos

    def generateSINKandSOURCE(self):
        """
		增加 sink 和 source，并更新邻接矩阵元素个数(+2)，并添加 sink 和 source 的邻居
		"""
        self.sink = self.G.nodeNumber
        self.source = self.G.nodeNumber + 1
        sinkNode = cpsNode(identity=self.sink, position=self.sink_pos, energy=self.G.initEnergy * 100,
                           radius=self.G.radius, state='SINK', adj=set())
        sourceNode = cpsNode(identity=self.source, position=self.source_pos, energy=self.G.initEnergy * 100,
                             radius=self.G.radius, state='SOURCE', adj=set())
        num = self.G.nodeNumber
        self.G.addNode(sinkNode)
        self.G.addNode(sourceNode)
        if self.G.nodeNumber != num + 2:
            print "Error in the deployed sink and source."
            while len(self.G.nodeList) > num:
                self.G.nodeList.pop()
        else:
            for i in self.G.nodeList[self.sink].adj:
                self.G.nodeList[i].adj.add(self.sink)
            for i in self.G.nodeList[self.source].adj:
                self.G.nodeList[i].adj.add(self.source)  #

    def deployAttacker(self, Node):
        """
		部署攻击者初始位置，位于 sink
		"""
        self.attacker.initDeploy(Node)

    def generateNetworkLevel(self):
        # print "构建网络层次"
        queue = Queue.Queue()
        visited = np.zeros(self.G.nodeNumber, dtype=np.int8)

        self.G.nodeList[self.sink].level = 0
        queue.put(self.sink)
        visited[self.sink] = 1
        # 计算节点与基站之间最短的跳数
        while not queue.empty():
            u = queue.get()
            visited[u] = 0
            for v in self.G.nodeList[u].adj:
                if self.G.nodeList[v].level == -1 or self.G.nodeList[u].level + 1 < self.G.nodeList[v].level:
                    self.G.nodeList[v].level = self.G.nodeList[u].level + 1
                    if visited[v] == 0:
                        queue.put(v)
                        visited[v] = 1

    def sendSource2Sink(self, Ti):

        packetSize = 500 * 8  # 单次数据包大小 bit
        # 路由时延 and 网络能耗
        flag = False
        delayTi = 0
        energyTi = 0
        # 延迟
        u = -1
        for i, v in enumerate(self.RW_set):
            if i == 0:
                u = v
            else:
                delayTi += self.G.delayModelusingAdjMatrix(u, v) * packetSize
                u = v
        for i, v in enumerate(self.MHR_path):
            if i == 0:
                u = v
            else:
                delayTi += self.G.delayModelusingAdjMatrix(u, v) * packetSize
                u = v

        # 网络能耗
        for node in self.G.nodeList:
            ec = self.G.energyModelusingAdjMatrix(node) * packetSize
            energyTi += ec
            self.G.nodeList[node.identity].energy -= ec
            if not self.G.nodeList[node.identity].isAlive():
                flag = True
        # 攻击者移动
        self.attacker.move(self.G)
        # 源节点是否被捕获
        if self.attacker.position.identity == self.source:
            print " "
            print "BE CAPTURE"
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
        each_result.append(len(self.RW_set) + len(self.MHR_path) - 1)
        # number of transmitted and broadcast fake messages
        each_result.append(np.max(self.G.adjacentMatrix, axis=1).sum() - len(self.RW_set) + len(self.MHR_path) + 1)
        self.result.append(each_result)

    def routing(self):
        """
        过程
        """
        listDelay = []
        listEnergyConsumption = []
        safety = -1
        for Ti in range(1, self.Tmax + 1):
            if Ti % 100 == 0:
                print Ti
            elif Ti % 10 == 0:
                print Ti,
            # start = datetime.datetime.now()
            self.num_branch_path = []
            self.MHR_path = []
            self.RW_set = []
            print " "
            print "the", Ti, 'step'
            a = self.RW_scheme(5, self.G.nodeList[self.source].identity)
            self.MHR_scheme(self.sink, a)
            # 源节点发送消息给基站的事件
            flag, delayTi, energyTi = self.sendSource2Sink(Ti)
            self.updateAdjMatrix()
            # 保存每轮的记录
            listDelay.append(delayTi)
            listEnergyConsumption.append(energyTi)
            #             end = datetime.datetime.now()
            #             print '\nIt costs', end-start
            if Ti == self.Tmax:
                safety = Ti
                print "到达阈值"
                break
            if flag:
                safety = Ti
                print "被捕获"
                break
        return safety, listDelay, listEnergyConsumption

    def RW_scheme(self, H, Ni):
        """
        算法1: 随机行走方案 (节点Ni)
        输入:
        H: 随机行走阶段的跳数；
        Ni: 源节点；
        输出:
        Cnode: 中间节点；
        随机行走阶段的路由路径；
        1: 初始化: Ni_Next_hop = null;
        2: 将节点Ni放入随机行走集合中;
        3: 当节点Ni接收到配置消息时执行以下操作:
        4: 构建候选集合;
        5: 从候选集合中删除随机行走节点;
        6: 从候选集合中选择剩余能量最多的节点作为下一跳节点Ni+1;
        7: H = H - 1;
        8: 如果 H == 0
        9: 返回;
        10: 否则执行随机行走方案 (Node Ni+1);
        11: 结束循环
        """
        self.Ni_Next_hop = None
        self.RW_set = [Ni]
        self.sum_path = [Ni]
        max = 1
        print "构建RW骨干网络..."
        queue = Queue.Queue()  # 创建一个队列对象，用于辅助进行广度优先搜索
        visited = np.zeros(self.G.nodeNumber, dtype=np.int8)  # 初始化一个数组 visited 用于标记节点是否被访问过。
        queue.put(Ni)
        visited[Ni] = 1
        # 计算节点与基站之间最短打跳数
        while not queue.empty():  #
            u = queue.get()
            visited[u] = 0
            ES = []
            CS = []
            FS = []
            candidate_set = []
            if u:
                for i in self.G.nodeList[u].adj:
                    if self.G.nodeList[i].level == self.G.nodeList[Ni].level and i not in self.RW_set:
                        ES.append(i)
                        candidate_set.append(i)
                    elif self.G.nodeList[i].level <= self.G.nodeList[Ni].level and i not in self.RW_set:
                        CS.append(i)
                    elif self.G.nodeList[i].level >= self.G.nodeList[Ni].level and i not in self.RW_set:
                        FS.append(i)
                        candidate_set.append(i)
                for i in candidate_set:
                    if self.G.nodeList[i].energy >= max:
                        max = self.G.nodeList[i].energy
                        self.Ni_Next_hop = i

                if visited[self.Ni_Next_hop] == 0 and self.Ni_Next_hop not in self.RW_set:
                    queue.put(self.Ni_Next_hop)
                    visited[self.Ni_Next_hop] = 1
                    self.RW_set.append(self.Ni_Next_hop)
                    self.sum_path.append(self.Ni_Next_hop)

                H -= 1
                if H == 0:
                    print self.RW_set
                    print "RW骨干构建完成"
                    return self.Ni_Next_hop
            else:
                print "u is none"
        return self.Ni_Next_hop

    def selact_node(self, node, source):
        slect_node = []
        c = None
        max = 1
        CS = []
        for i in self.G.nodeList[node].adj:
            if self.G.nodeList[i].level <= self.G.nodeList[node].level:
                if i not in self.RW_set and i not in self.MHR_path:
                    # vec_x = self.G.nodeList[i].position[0] - self.G.nodeList[source].position[0]
                    # vec_y = self.G.nodeList[i].position[1] - self.G.nodeList[source].position[1]
                    # angle = math.degrees(math.atan2(vec_y, vec_x))
                    # if 135 < angle < 180 or -180 < angle < -45:
                    #     angle = 360 + angle
                    # a = angle
                    # if 135 < a < 315:
                    CS.append(i)
        if CS:
            for v in CS:
                if self.G.nodeList[v].energy > max:
                    max = self.G.nodeList[v].energy
                    c = v
            return c
        else:
            print "没有找到点"
        # return c

    def MHR_scheme(self, sink_node, Ni):
        """
        输入：
        汇聚节点（sink node）
        中间节点（Ni）
        输出：
        MHR阶段的路由路径（routing path）
        分支路径（branch path）

        初始化：Ni_Next_hop = null（下一跳节点初始值为空）
        获取可见区域角度（α1, α3）
        获取随机行走角度（α2, α4）
        α的范围 = min(α1, α2) 到 max(α3, α4)
        当 Ni 不等于汇聚节点时执行以下循环：
        构建候选集合，并从候选集合中删除随机行走节点
        从候选集合中选择剩余能量最多的节点作为下一跳节点，并且该节点不在α的范围内
        如果 Ni 等于 Cnode：
        执行Bran0(Cnode, hop)
        否则：
        生成一个0到1之间均匀分布的随机数Ran
        如果 Ran 小于 p：
        执行Bran(Ni, hop)
        将下一跳节点作为当前节点Ni
        结束循环

        :param Ni:
        :return:
        """

        # def get_visible_area_angle():
        #     # 获取可见区域角度 (α1, α3)
        #     pass
        #
        # def get_RW_angle():
        #     # 获取随机行走角度 (α2, α4)
        #     pass
        #
        # def select_next_hop(candidate_set, angle):
        #     # 从候选集合中选择剩余能量最多的节点作为下一跳节点，且不在角度范围内
        #     pass
        #
        # def Bran(node, hop):
        #     # 分支方法
        #     pass

        # Ni_Next_hop = None
        self.MHR_path = [Ni]
        self.branchi_path = []
        cnode = Ni
        w = 0.5
        hop = 5
        t = 1
        zt = 1
        print "构建MHR骨干网络..."
        while Ni != sink_node:
            a = self.G.nodeList[self.Ni_Next_hop].level - self.G.nodeList[Ni].level
            b = a / self.G.nodeList[cnode].level
            p = (1 - b) * w
            hop1 = int(hop / zt + math.log(self.G.nodeList[Ni].level))
            if t == 1:
                self.Bran0_scheme(Ni, hop)
                t = 2
                # Ni_Next_hop = None
            else:
                Ran = random.uniform(0, 1)
                if Ran < 0.2:  # p
                    # 步骤13: Bran(Ni, hop)
                    b = self.Bran_scheme(Ni, hop1)
                    self.branchi_path.append(b)

            Ni = self.selact_node(Ni, self.source)
            if Ni:
                # 将当前节点添加到路由路径中
                if Ni not in self.MHR_path:
                    self.MHR_path.append(Ni)
                    self.sum_path.append(Ni)

                if Ni == sink_node:
                    print self.MHR_path
                    print "MHR骨干网络构建完成"
                    print "总路线为：", self.sum_path
            else:
                print "Ni is NONE"
        return self.MHR_path

    def Bran0_scheme(self, Ni, H):
        """
        输入：
        中间节点（Ni）
        Bran0的长度（H0）
        输出：
        Bran0阶段的路由路径

        初始化：Ni_Next_hop = null（下一跳节点初始值为空）
        当接收到Bran0消息时执行以下循环：
        构建候选集合
        从候选集合中删除随机行走节点和Bran0节点
        从候选集合中选择剩余能量最多的节点作为下一跳节点
        H0 = H0 - 1
        如果 H0 等于 0：
        返回结果
        结束判断
        结束循环
        :param H0:
        :return:
        """
        Ni_Next_hop = None
        self.bron0_path = [Ni]
        max = 1
        print "构建bron0分支..."
        queue = Queue.Queue()  # 创建一个队列对象，用于辅助进行广度优先搜索
        visited = np.zeros(self.G.nodeNumber, dtype=np.int8)  # 初始化一个数组 visited 用于标记节点是否被访问过。
        queue.put(Ni)
        visited[Ni] = 1
        # 计算节点与基站之间最短打跳数
        while not queue.empty():  #
            u = queue.get()
            visited[u] = 0
            FS = []
            candidate_set = []
            for i in self.G.nodeList[u].adj:
                if self.G.nodeList[i].level >= self.G.nodeList[Ni].level:
                    if i not in self.RW_set and i not in self.MHR_path and i not in self.bron0_path:
                        FS.append(i)
                        candidate_set.append(i)
            for i in candidate_set:
                if self.G.nodeList[i].energy >= max:
                    max = self.G.nodeList[i].energy
                    Ni_Next_hop = i

            if visited[Ni_Next_hop] == 0:
                queue.put(Ni_Next_hop)
                visited[Ni_Next_hop] = 1
                self.bron0_path.append(Ni_Next_hop)

            H -= 1
            if H == 0:
                print self.bron0_path
                # print "bron0分支构建完成"
                return Ni_Next_hop
        return Ni_Next_hop

    def Bran_scheme(self, Ni, H):
        """
        它的输入是节点 Ni（位于 MHR 路径中的节点）和 Brani 的长度 Hi，
        输出是 Brani 的路由路径。
        算法步骤的解释如下：
        初始化：将 Ni 的下一跳节点设为 null。
        当接收到 Bran-i 消息时执行循环。
        构建候选集合：根据算法要求构建候选集合。
        从候选集合中删除转发节点、分支节点和中心节点。
        获取分支角度 θ。
        从候选集合中选择剩余能量最多的节点，并且角度在 γ 的范围内作为下一跳节点。
        Hi 减 1。
        如果 Hi 等于 0，则结束循环。
        返回 Brani 的路由路径。
        结束循环。
        :param Hi:
        :return:
        """
        Ni_Next_hop = None
        self.broni_path = [Ni]
        max = 1
        print "构建broni分支..."
        queue = Queue.Queue()  # 创建一个队列对象，用于辅助进行广度优先搜索
        visited = np.zeros(self.G.nodeNumber, dtype=np.int8)  # 初始化一个数组 visited 用于标记节点是否被访问过。
        queue.put(Ni)
        visited[Ni] = 1
        if Ni:
            m = 1
        else:
            print "Ni is none branchi not start"
        while not queue.empty():  #
            u = queue.get()
            visited[u] = 0
            FS = []
            ES = []
            candidate_set = []
            for i in self.G.nodeList[u].adj:
                if self.G.nodeList[i].level == self.G.nodeList[Ni].level:
                    if i not in self.RW_set and i not in self.MHR_path and i not in self.bron0_path and i not in self.num_branch_path:
                        ES.append(i)
                        candidate_set.append(i)
                if self.G.nodeList[i].level >= self.G.nodeList[Ni].level:
                    if i not in self.RW_set and i not in self.MHR_path and i not in self.bron0_path and i not in self.num_branch_path:
                        FS.append(i)
                        candidate_set.append(i)
                vec_x = self.G.nodeList[i].position[0] - self.G.nodeList[u].position[0]
                vec_y = self.G.nodeList[i].position[1] - self.G.nodeList[u].position[1]
                angle = math.degrees(math.atan2(vec_y, vec_x))
            if candidate_set:
                for i in candidate_set:
                    if i not in self.RW_set and i not in self.MHR_path and i not in self.bron0_path and i not in self.sum_path:
                        if self.G.nodeList[i].energy >= max:
                            max = self.G.nodeList[i].energy
                            Ni_Next_hop = i
                if Ni_Next_hop:
                    if visited[Ni_Next_hop] == 0:
                        queue.put(Ni_Next_hop)
                        visited[Ni_Next_hop] = 1
                        self.broni_path.append(Ni_Next_hop)
                else:
                    print "branchi 里面的 Ni_Next_hop 是NONE"
            else:
                print "branchi 里面的 candidate_set 是NONE"

            self.num_branch_path.append(Ni_Next_hop)

            H -= 1
            if H == 0:
                print self.broni_path
                # print "broni分支构建完成"
                # print "总路线为：", self.sum_path
                return self.broni_path
        return self.broni_path

    def updateAdjMatrix(self):
        self.G.initAdjMatrix()
        for (u, v) in zip(self.RW_set[:-1], self.RW_set[1:]):
            self.G.adjacentMatrix[u, v] = 1
        for (q, w) in zip(self.bron0_path[:-1], self.bron0_path[1:]):
            self.G.adjacentMatrix[q, w] = 1
        for (e, r) in zip(self.MHR_path[:-1], self.MHR_path[1:]):
            self.G.adjacentMatrix[e, r] = 1
        for i in self.branchi_path:
            i = i[::-1]
            for (t, y) in zip(i[:-1], i[1:]):
                self.G.adjacentMatrix[t, y] = 1
        for i in self.branchi_path:
            for (h, j) in zip(i[:-1], i[1:]):
                self.G.adjacentMatrix[h, j] = 1

    def rwplot(self):
        """
         b'：蓝色（blue）'g'：绿色（green）'r'：红色（red）'c'：青色（cyan）'m'：洋红色（magenta）'y'：黄色（yellow）
         'k'：黑色（black）'w'：白色（white）
         '-'：实线 '--'：虚线'-.'：点划线':'：点线'.'：点','：像素'o'：圆圈'^'：上三角形'v'：下三角形'<'：左三角形
         '>'：右三角形's'：正方形'+'：加号'*'：星号'x'：叉号'D'：菱形
        """
        ax = plt.figure(figsize=(20, 20))
        nom_x = []
        nom_y = []
        for i in range(self.G.nodeNumber):
            if i in self.RW_set and self.MHR_path and self.bron0_path:
                continue
            nom_x.append(self.G.nodeList[i].position[0])
            nom_y.append(self.G.nodeList[i].position[1])
        plt.plot(nom_x, nom_y, 'ko')
        # RW骨干
        u = -1
        for i, v in enumerate(self.RW_set):
            if i == 0:
                u = v
                continue
            else:
                U = self.G.nodeList[u]
                V = self.G.nodeList[v]
                x = [U.position[0], V.position[0]]
                y = [U.position[1], V.position[1]]
                plt.plot(x, y, 'k')  # 绘制两点之间连线
                u = v
        # MHR骨干
        m = -1
        for i, v in enumerate(self.MHR_path):
            if i == 0:
                m = v
                continue
            else:
                U = self.G.nodeList[m]
                V = self.G.nodeList[v]
                x = [U.position[0], V.position[0]]
                y = [U.position[1], V.position[1]]
                plt.plot(x, y, 'k')  # 绘制两点之间连线
                m = v
        # bran0
        o = -1
        for i, v in enumerate(self.bron0_path):
            if i == 0:
                o = v
                continue
            else:
                U = self.G.nodeList[o]
                V = self.G.nodeList[v]
                x = [U.position[0], V.position[0]]
                y = [U.position[1], V.position[1]]
                plt.plot(x, y, 'g')  # 绘制两点之间连线
                o = v

        # brani
        for w in self.branchi_path:
            e = -1
            for i, v in enumerate(w):
                if i == 0:
                    e = v
                    continue
                else:
                    U = self.G.nodeList[e]
                    V = self.G.nodeList[v]
                    x = [U.position[0], V.position[0]]
                    y = [U.position[1], V.position[1]]
                    plt.plot(x, y, 'r')  # 绘制两点之间连线
                    e = v
        # RW节点
        temp_x = []
        temp_y = []
        for i in self.RW_set:
            temp_x.append(self.G.nodeList[i].position[0])
            temp_y.append(self.G.nodeList[i].position[1])
        plt.plot(temp_x, temp_y, 'ro')  # ro红圆ko黑圆wo白圆rs红方
        # MHR节点
        mhr_x = []
        mhr_y = []
        for i in self.MHR_path:
            mhr_x.append(self.G.nodeList[i].position[0])
            mhr_y.append(self.G.nodeList[i].position[1])
        plt.plot(mhr_x, mhr_y, 'mo')
        # bran0
        bran_x = []
        bran_y = []
        for i in self.bron0_path:
            bran_x.append(self.G.nodeList[i].position[0])
            bran_y.append(self.G.nodeList[i].position[1])
        plt.plot(bran_x, bran_y, 'ro')
        # brani
        for v in self.branchi_path:
            brani_x = []
            brani_y = []
            for i in v:
                brani_x.append(self.G.nodeList[i].position[0])
                brani_y.append(self.G.nodeList[i].position[1])
            plt.plot(brani_x, brani_y, 'ro')

        plt.axis("equal")
        plt.show()

    def multi(self):
        sum_delay = 0
        self.generateSINKandSOURCE()  # 部署 sink 和 source
        self.deployAttacker(self.G.nodeList[self.sink])  # 部署攻击者
        self.generateNetworkLevel()  # 生成网络层次
        # self.RW_scheme(5, self.G.nodeList[self.source].identity)
        # self.MHR_scheme(self.G.nodeList[self.sink].identity, self.Ni_Next_hop)
        # self.updateAdjMatrix()
        self.safety, self.listDelay, self.listEnergyConsumption = self.routing()
        self.rwplot()
        for i in range(len(self.listDelay)):
            sum_delay += self.listDelay[i]
            mean_delay = sum_delay / (i + 1)
        print "\nThe safety is", self.safety, "\nThe every listDelay is", self.listDelay, "\nThe SumDelay is", sum_delay, "\nThe MeanDelay is", mean_delay


if __name__ == '__main__':
    network = cpsNetwork(file_path='load_network/temp_network.csv')
    print '网络规模：', network.nodeNumber, network.areaLength
    tb = multiroad(G=network,
                   Tmax=2000, Hwalk=10,
                   sink_pos=(-200, -200), source_pos=(200, 200))
    # tb.display()

    tb.multi()
    restEnergy = [tb.G.initEnergy - node.energy for node in tb.G.nodeList if
                  node.identity != tb.source and node.identity != tb.sink]
    # print restEnergy
    print "\nThe maxrestEnergy is", max(restEnergy), "\nThe neanrestEnergy is", np.mean(
        restEnergy), "\nThe minrestEnergy is", min(restEnergy), "\nThe stdrestEnergy is", np.std(restEnergy)
    # 最大值、平均值、最小值和标准差
    # print np.array(tb.result)
    # tb.attacker.display()
    tb.attacker.display()

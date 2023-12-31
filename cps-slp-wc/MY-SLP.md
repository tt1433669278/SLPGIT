# MY-SLP

## 背景：

工控环境，工业世界中的能源网络中（像电网之类的）【具体的环境】。每个设备可以看成一个节点，因此，如果任何节点、物理设备或控制器被恶意软件感染，攻击者就可以访问当前的网络

解决问题：提高安全周期的同时降低能耗和延迟

具体

源节点和汇聚节点的确定-----------虚假节点的选择（分层分区）-------------骨干网络构建（启发）--------------动态虚假源节点的选择（骨干网络附近能源高的节点）



## CPS Network Model

### 攻击模型

<img src="C:\Users\14336\AppData\Roaming\Typora\typora-user-images\image-20230810200003081.png" alt="image-20230810200003081" style="zoom:80%;" />



## 对比指标

延迟：![image-20230625143452577](C:\Users\14336\AppData\Roaming\Typora\typora-user-images\image-20230625143452577.png)

骨干网络上的每个节点间的延迟，再叠加

![image-20230625143749322](C:\Users\14336\AppData\Roaming\Typora\typora-user-images\image-20230625143749322.png)

节点消耗：

接受和发送的能耗总和，与距离关联。

安全期：

捕获率:



## 具体路径

源节点和汇聚节点的确定-----------幻影节点的选择（分区）-------------骨干网络构建（启发）--------------虚假源节点的选择（骨干网络附近能源高的节点）

### 幻影节点的选择：

![image-20230810201016906](C:\Users\14336\AppData\Roaming\Typora\typora-user-images\image-20230810201016906.png)



源节点和汇聚节点跳数之间选择幻影节点，但要规避可视区域，一旦攻击者回溯到可视区域内，源节点就会被捕获，经过可视区的传输路径被称作失效路径。以源节点为圆心，攻击者感知范围为半径，构建可视区域。因为骨干路径我选择是每一次的最短路径，所以不能选择中间的节点。

橙色为幻影节点，之后对所选的幻影节点进行分区，每一轮让他们交替充当幻影节点，之后根据幻影节点进行骨干路径的建设

<img src="C:\Users\14336\AppData\Roaming\Typora\typora-user-images\image-20230810194839621.png" alt="image-20230810194839621" style="zoom:80%;" />

### 骨干网络的构建：

广度优先，启发式搜索，本文章的搜索算法是针对单个目标搜索进行的优化，

开始搜索：

1. 从起点 A 开始，并把它就加入到一个 open list( 开放列表 ) 中。这个 open list 有点像是一个购物单。当然现在 open list 里只有一项，它就是起点 A ，后面会慢慢加入更多的项。 Open list 里的内容是路径可能会是沿途经过的，也有可能不经过。基本上 open list 是一个待检查的方格列表。

2. 查看与起点 A 的邻居节点 ，把其中可走的 (walkable) 或可到达的 (reachable) 方格也加入到 open list 中。把起点 A 设置为这些方格的父亲 (parent node 或 parent square) 。当我们在追踪路径时，这些父节点的内容是很重要的。

3. 把 A 从 open list 中移除，加入到 close list( 封闭列表 ) 中， close list 中的每个方格都是现在不需要再关注的。

路径排序：

用当前节点x的一个函数f(x),当前节点x的函数定义：
$$
f (x)= g(x)+h(x)
$$


- f(n)是节点n的综合优先级。当我们选择下一个要遍历的节点时，我们总会选取综合优先级最高（值最小）的节点。
- g(n) 是节点n距离起点的代价，使用欧几里得距离。

​       
$$
G = w*d
$$
这里的w是代价因子根据环境的不同选取不同的值，如给平地地形设置代价因子为1，丘陵地形为2，在移动代价相同情况下，平地地形的G值更低，算法就会倾向选择G值更小的平地地形，在这里环境统一视为平地。

- h(n)是节点n距离终点的预计代价，用欧几里得距离。

- 欧几里得距离是指两个节点之间的直线距离，因此其计算方法也是我们比较熟悉的：

  

  ![img](https://pic1.zhimg.com/80/v2-1f142f9e75823c1ec34f83f65d723470_720w.webp)

算法基本实现过程为：从起始点开始计算其每一个子节点的f值，从中选择f值最小的子节点作为搜索的下一点，并将当前节点作为下一节点的父节点，往复迭代，直到下一子节点为目标点，最后再通过目标节点的父节点回溯到起点。

### 虚假节点的选择：

根据节点剩余能量，和他与邻居节点的平均距离来判断，因为选择虚假节点会增加网络安全性，但是过度的使用虚假节点会减少节点的能量，从而会减少安全性，而距离太远也会增加能耗和延迟，故根据这两个因素来选择每一轮的节点作为虚假节点。

![image-20230810203706409](C:\Users\14336\AppData\Roaming\Typora\typora-user-images\image-20230810203706409.png)

其中numb是附近节点中是骨干节点的个数

numi邻居节点个数

Enode节点当前能量

Einit 节点初始能量

dist是当前节点与邻居节点距离累加

node——dist是节点与源节点距离
$$
X=MAX [\frac{Eres}{Einit} ,NUM ]
$$

## 理论

### 接近中心性（Closeness centrality）



接近中心性（closeness centrality）度量的思想是：如果一个节点与许多其他节点都很“接近”（close），那么节点处于网络中心位置（central）。根据Sabidussi描述的标准计算方法，这一中心性定义为某节点到其它所有节点距离之和的倒数即：

![img](https://www.sci666.com.cn/wp-content/uploads/2020/02/wxsync-2020-02-5f402bea26153c1f29650ef0a19668e3.png)

改：分权重
$$
C=w1*N+w2*dist(v ,u)
$$
其中dist(v,u)是节点u,v∈V的捷径距离。通常这一度量会乘以系数Nv-1归一化到[0,1]区间，用于不同图之间以及不同中心性度量之间的比较。

因为攻击者有1/24的概率通过在骨干中随机选择一个具有等概率的中继节点来捕获源。

设想一个实际生活中的场景，比如要建一个大型的娱乐商场，设计者可能会希望周围的顾客到达这个商场的距离都可以尽可能地短，这个就涉及到接近中心性的概念。在网络中，接近中心性它反映了网络中某一节点与其他节点之间的接近程度。即对于一个节点，它距离其他节点越近，周围节点越多，那么它的接近性中心性越大，也就越“重要”。

1. 影响力：节点的邻居节点数量可以反映其在网络中的影响力。如果一个节点有很多邻居节点，它的消息、影响或传播能力可能更强。这是因为邻居节点的数量越多，信息传播和影响传播的潜在范围就越大。
2. 可达性：邻居节点的增多可以提高节点的可达性。如果一个节点有更多的邻居节点，它可以更快地到达其他节点，从而促进信息传递、资源交换或协作行为。
3. 多样性：邻居节点的增多意味着节点在网络中接触到更多不同的信息、观点和资源。这可以增加节点的多样性和知识范围，提供更多的选择和机会。

节点剩余能量，上一次是否为虚假节点，虚假节点的选择：

节点中心性+能源消耗+上一次是否是虚假源
$$
CE = (1 - RANKE/N) +CP
$$
RANKE：节点在周围节点中能源排名

CP：上一次是否参与过虚假源的计算
$$
P=exp(numb/N)/(C+CE)
$$
numb：邻居节点是骨干节点



幻影节点，选择

## 开会

两篇近几年论文

设计一个总目标函数



理论挖掘，根据算法扩展理论证明

实验参数，根据参数的不同设计不同的实验

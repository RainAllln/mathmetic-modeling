#迪杰斯特拉算法
# the graph 绘制图
graph1 = {}
graph1["start"] = {}
graph1["start"]["a"] = 6
graph1["start"]["b"] = 2

graph1["a"] = {}
graph1["a"]["fin"] = 1

graph1["b"] = {}
graph1["b"]["a"] = 3
graph1["b"]["fin"] = 5

graph1["fin"] = {}

# the costs table 成本表
infinity = float("inf")
costs = {}
costs["a"] = 6
costs["b"] = 2
costs["fin"] = infinity

# the parents table 父节点表
parents = {}
parents["a"] = "start"
parents["b"] = "start"
parents["fin"] = None

processed = []

def find_lowest_cost_node(costs):
    lowest_cost = float("inf")
    lowest_cost_node = None
    # 遍历每个节点。
    for node in costs:
        cost = costs[node]
        # 如果这是目前为止最低的成本而且还没有被处理......
        if cost < lowest_cost and node not in processed:
            # ......设置为新的最低成本节点。
            lowest_cost = cost
            lowest_cost_node = node
    return lowest_cost_node

# 查找尚未处理的最低成本节点。
node = find_lowest_cost_node(costs)
# 如果已经处理了所有节点，那么while循环就完成了。
while node is not None:
    cost = costs[node]
    # 遍历此节点的所有邻居。
    neighbors = graph[node]
    for n in neighbors.keys():
        new_cost = cost + neighbors[n]
        # 如果通过这个节点到这个邻居比较便宜的话......
        if costs[n] > new_cost:
            # ......更新此节点的成本。
            costs[n] = new_cost
            # 此节点将成为此邻居的新父节点。
            parents[n] = node
    # 将节点标记为已处理。
    processed.append(node)
    # 找到下一个要处理的节点，然后循环。
    node = find_lowest_cost_node(costs)

print("Cost from the start to each node:")
print(costs)

#弗洛伊德算法

INF = float('inf')

def floyd_warshall(graph2):
    # 初始化距离矩阵
    dist = [[INF if i != j else 0 for j in range(len(graph2))] for i in range(len(graph2))]

    # 更新距离矩阵
    for i in range(len(graph2)):
        for j in range(len(graph2)):
            if graph2[i][j] != 0:  # 如果节点i和节点j之间有直接连接的边
                dist[i][j] = graph2[i][j]

    # 动态规划更新距离矩阵
    for k in range(len(graph2)):
        for i in range(len(graph2)):
            for j in range(len(graph2)):
                if dist[i][k] != INF and dist[k][j] != INF:  # 如果节点i到k和k到节点j之间有路径
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


# 示例图的邻接矩阵表示
graph2 = [
    [0, 5, INF, 10],
    [INF, 0, 3, INF],
    [INF, INF, 0, 1],
    [INF, INF, INF, 0]
]

# 打印最短路径距离矩阵
result = floyd_warshall(graph2)
for row in result:
    print(row)

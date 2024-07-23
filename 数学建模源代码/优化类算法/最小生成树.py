#Prim算法
from collections import defaultdict
from heapq import heapify, heappush, heappop
import time
start = time.perf_counter()
def Prim(nodes, edges):
    '''  基于最小堆实现的Prim算法  '''
    element = defaultdict(list)
    for start, stop, weight in edges:
        element[start].append((weight, start, stop))
        element[stop].append((weight, stop, start))
    all_nodes = set(nodes)
    used_nodes = set(nodes[0])
    usable_edges = element[nodes[0]][:]
    heapify(usable_edges)
    # 建立最小堆
    MST = []
    while usable_edges and (all_nodes - used_nodes):
        weight, start, stop = heappop(usable_edges)
        if stop not in used_nodes:
            used_nodes.add(stop)
            MST.append((start, stop, weight))
            for member in element[stop]:
                if member[2] not in used_nodes:
                    heappush(usable_edges, member)

    return MST

def main():
    nodes = list('123456789')
    edges = [("1", "2", 5), ("1", "3",13),("1", "4",12), ("1", "5",10),
                   ("1", "6", 8), ("1", "7", 6),("1", "8", 2), ("1", "9", 5),
                   ("2", "3", 3), ("2", "9", 1),("3", "4", 9), ("4", "5",11),
                   ("5", "6", 9), ("6", "7", 6),("7", "8", 7), ("8", "9", 4)]
    print("\n\nThe undirected graph is :", edges)
    print("\n\nThe minimum spanning tree by Prim is : ")
    print(Prim(nodes, edges))

if __name__ == '__main__':
    main()
end = time.perf_counter()
print('Running time: %f seconds'%(end-start))

#克鲁斯卡尔算法

def find_parent(parent, node):
    if parent[node] != node:
        parent[node] = find_parent(parent, parent[node])
    return parent[node]


def union(parent, node1, node2):
    parent[node1] = node2


def kruskal(edges, vertices):
    parent = {node: node for node in range(vertices)}
    minimum_spanning_tree = []

    # Sort the edges based on their weight
    edges.sort(key=lambda edge: edge[2])

    for edge in edges:
        node1, node2, weight = edge
        root1 = find_parent(parent, node1)
        root2 = find_parent(parent, node2)

        # Check if the two nodes are not part of the same set
        if root1 != root2:
            minimum_spanning_tree.append(edge)
            union(parent, root1, root2)

    return minimum_spanning_tree


# Example usage:
vertices = 10 #节点数+1
edges = [(1, 2, 5), (1, 3, 13), (1, 4, 12), (1, 5, 10),
         (1, 6, 8), (1, 7, 6), (1, 8, 2), (1, 9, 5),
         (2, 3, 3), (2, 9, 1), (3, 4, 9), (4, 5, 11),
         (5, 6, 9), (6, 7, 6), (7, 8, 7), (8, 9, 4)]

print(kruskal(edges, vertices))
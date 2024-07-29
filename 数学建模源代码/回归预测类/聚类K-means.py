# 原理:通过迭代的方式将数据点分配到K个簇中，使得每个数据点到其所属簇的中心点(质心)的距离最小化。
# 把n个点(可以是样本的一次观察或一个实例)划分到k个集群(custer)，使得每个点都属于离他最近的均值(即聚类中心，centroid)对应的集群。
# 重复上述过程一直持续到重心不改变。
# 精度比KNN好，但是速度慢一点，而且K-means会实时更改中心位置
# https://blog.csdn.net/fengdu78/article/details/131746067?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172225539216800175710896%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172225539216800175710896&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-131746067-null-null.142^v100^pc_search_result_base1&utm_term=%E8%81%9A%E7%B1%BB&spm=1018.2226.3001.4187
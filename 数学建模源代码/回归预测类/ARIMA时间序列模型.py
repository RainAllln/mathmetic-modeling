# ARIMA模型
# 时间序列:按照时间排序的数值序列，分为两类
# 1.时期序列中，数值要素反映一定时期内的发展结果，比如中国历年GDP
# 2.时点序列中，数值要素反映一定时点的瞬时水平，比如每年测一次身高体重
# 其中时期序列可加，时点序列不可加
# 相加结果表明现象在更长一段时间内的活动总量
#
# 差分整合移动平均自回归模型ARIMA(p,q,d)，其中包含自回归移动平均模型ARMA(p,q)和差分I(d),ARMA(p,q)包含自回归模型AR(p),移动平均模型MA(q)
#
# 自回归模型AR(p)
# 用变量自身的历史数据对自身预测，其必须满足平稳性要求，只适用于预测与自身前期相关的现象
# p阶自回归过程定义:y_t=μ+Σ(i=1→p)γ_i*y_(t-i)+ε_t
# 其中p表示用几期的历史值预测，y_t是当前值，μ是常数项，p是阶数，γ_i是自相关系数
#
# 移动平均模型MA(q)
# 关注的是误差项累计
# q阶公式定义:y_t=μ+ε_t+Σ(i=1→q)θ_i*ε_(t-i)
#
# 结合得ARMA(p,q)
# y_t=μ+Σ(i=1→p)γ_i*y_(t-i)+ε_t+Σ(i=1→q)θ_i*ε_(t-i)
#
# ARIMA(p,q,d)
# p是自回归项，q为平均移动次数，d为时间序列成为平稳时的差分次数
# 原理:将非平稳时间序列转化成平稳时间序列然后将因变量仅对他的滞后值以及随机误差项的现值进行回归
#
# 基本步骤:
# 1.对序列绘图，进行平稳性检验，观察序列是否平稳，非平稳序列需要进行d阶差分转化
# 2.对平稳序列求自相关系数(ACF)和偏自相关系数(PACF),通过对自相关图和偏自相关图分析或通过AIC/BIC搜索，得到最佳阶p,q
# 3.通过p,q,d得到最佳模型，检验模型
#
# 平稳性
# 要求经由样本时间序列所得到的拟合曲线在未来一段时间内仍然能够按照现有的形态延续下去
# 要求序列的均值和方差不发生明显变化
#
# 差分法
# 时间序列在t和t-1时刻的差值，将非平稳序列变平稳
# △yx=y(x+1)-y(x)
# 比如[0,1,2,3,4]差分后[1,1,1,1]
#
# 自相关系数(ACF)
# 自相关系数反映了统一序列在不同时序的取值之间的相关性
# y_t与y_(t-k)的相关系数称为y_t间隔k的自相关系数
# 公式ACF(k)=ρ_k=Cov(y_t,y_(t-k))/Var(y_t)取值范围为[-1,1]
#
# 偏自相关系数(PACF)
# 为了能单纯测度x(t-k)对x(t)的影响，引进偏自相关系数(PACF)的概念。
# 对于平稳时间序列{x(t)}，所谓滞后k偏自相关系数指在剔除了中间k-1个随机变量x(t-1),x(t-2),.. ,x(t-k+1)的干扰之后，x(t-k)对x(t)影响的相关程度。
# 公式PACF(k)=Cov[(Z_t-Z_t^-),(Z_(t-k),Z_(t-k)^-)]/√(var(Z_t-Z_t^-))*√(var(Z_(t-k)-Z_(t-k)^-))
#
# ADF检验
# ADF大致的思想就是基于随即游走(不平稳的一个特殊序列)的，对其进行回归，如果发现p=1，说明序列满足随机游走，就是非平稳的
#
# 截尾和拖尾
# 截尾(出现以下情况，通常视为(偏)自相关系数d阶截尾)
# 1)在最初的d阶明显大于2倍标准差范围2)之后几乎95%的(偏)自相关系数都落在2倍标准差范围以内
# 3)且由非零自相关系数衰减为在零附近小值波动的过程非常突然
# 拖尾(出现以下情况，通常视为(偏)自相关系数拖尾)
# 1)如果有超过5%的样本(偏)自相关系数都落入2倍标准差范围之外
# 2)或者是由显著非0的(偏)自相关系数衰减为小值波动的过程比较缓慢或非常连续
#
#             ACF拖尾               ACF截尾
#  PACF拖尾   ARMA(p,q)             MA(q)
#  PACF截尾     AR(p)           ARMA类模型不适用
#
# 利用BIC准则确定p,q值，指标越小越好
# BIC=-2*ln(L)+K*ln(n) ,n表示样本容量,K表示参数个数，L表示极大似然函数
# 通过网格化搜索确定p,q,用AR0取到AR5和MA0到MA5，之后两两组合成5*5矩阵，找到其中最小的BIC值，得到最佳p,q
import pandas as pd
import matplotlib.pyplot as plt
#提取数据
ChinaBank = pd.read_csv("ChinaBank.scv",index_col='Date',parse_dates=['Date'])
ChinaBank.head()
#提取close列
ChinaBank.index = pd.to_datetime(ChinaBank.index)
sub= ChinaBank.loc['2014-01':"2014-06','close"]
sub.head()
#划分训练集
train = sub.loc['2014-01':'2014-03']
test = sub.loc['2014-04':'2014-06']
#画出训练集和测试集数据图
plt.figure(figsize=(12,6))
plt.plot(train)
plt.xticks(rotation=45)#x轴标记能转45度
plt.show()

#差分法
ChinaBank['diff_1'] = ChinaBank['close'].diff(1) #1阶差分
ChinaBank['diff_2'] = ChinaBank['diff_1'].diff(1) #2阶差分
fig=plt.figure(figsize=(12,10))#原数据
ax1 =fig.add_subplot(311)
ax1.plot(ChinaBank['close'])
ax2 =fig.add_subplot(312)
ax2.plot(ChinaBank['diff_1'])
ax3=fig.add_subplot(313)
ax3.plot(ChinaBank['diff_2'])
plt.show()

#ADF检验
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF
#计算原始序列、一阶差分序列、二阶差分序列的单位根检验结果
ChinaBank['diff_1'] = ChinaBank['diff_1'].fillna(0) #需要连续，所以填充缺失值
ChinaBank['diff 2'] = ChinaBank['diff 2'].fillna(0)
timeseries_adf = ADF(ChinaBank['close'].tolist())
timeseries_diff1_adf = ADF(ChinaBank['diff 1'].tolist())
timeseries_diff2_adf = ADF(ChinaBank['diff 2'].tolist())
#打印单位根检验结果
print('timeseries adf :',timeseries_adf)
print('timeseries diff1 adf :',timeseries_diff1_adf)
print('timeseries diff2 adf :',timeseries_diff2_adf)

#参数确定
import statsmodels.api as sm
#绘制
fig=plt.figure(figsize=(12,7))
ax1 = fig.add_subplot(211)
fig =sm.graphics.tsa.plot_acf(train, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')# 设置坐标轴上的数字显示的位置，top:显示在顶部bottom:显示在底部
# #fig.tight_layout()
ax2 =fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
#fig.tight_layout()
plt.show()

#模型建立
import itertools
import numpy as np
import seaborn as sns
p_min = 0
d_min = 0
q_min = 0
p_max = 5
q_max = 5
d_max = 5
#Initialize a DataFrame to store the results,,以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p),'MA{}'.format(q)]= np.nan
        continue
    try:
        model=sm.tsa.ARIMA(train,order=(p,d,q),
                           #enforce_stationarity=False,
                           # #enforce_invertibility=False,
                           )
        results = model.fit()
        results_bic.loc['AR{}'.format(p),'MA{}'.format(q)]= results.bic
    except:
        continue

#得到结果后进行浮点型转换
results_bic = results_bic[results_bic.columns].astype(float)

#绘制热力图
fig,ax=plt.subplots(figsize=(10,8))
ax= sns.heatmap(results_bic,mask=results_bic.isnull(),ax=ax,annot=True,fmt='.2f',cmap="Purples")
ax.set_title('BIC')
plt.show()

results_bic.stack().idxmin() #网格搜索得最小值

#提取p和q最优值
train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n',max_ar=8,max_ma=8)

print('AIC',train_results.aic_min_order)
print('BIC',train_results.bic_min_order)

#模型检验
p = 1 #根据BIC结果
d = 0
q = 0
model = sm.tsa.ARIMA(train,order=(p,d,q))
results = model.fit()
resid = results.resid #获取残差
#绘制
#查看测试集的时间序列与数据(只包含测试集)
fig, ax=plt.subplots(figsize=(12,5))
ax=sm.graphics.tsa.plot_acf(resid, lags=40,ax=ax)
plt.show()

#模型预测
predict_sunspots = results.predict(dynamic=False)
print(predict_sunspots)
#画图
#查看测试集的时间序列与数据(只包含测试集)
plt.figure(figsize=(12,6))
plt.plot(train)
plt.xticks(rotation=45)#旋转45度
plt.plot(predict_sunspots)
plt.show()
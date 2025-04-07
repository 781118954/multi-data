from matplotlib import pyplot as plt
import KOA
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

''' --------------------------- 参数设置 ----------------------------------'''

# 输入数据路径
data_path = 'mult.csv'  # 数据
data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=1)

# 划分数据标签
cols = data.shape[1]
X = data[:, :-1]
y = data[:, cols - 1:cols]
y = np.array(y.ravel())  # 数列平铺
y = np.squeeze(y)
print("X_filtered", X.shape)
data_D = X
data_L = y
X_train, X_test, y_train, y_test= train_test_split(data_D, data_L, random_state=1, test_size=0.2)

SearchAgents_no = 10 # 种群量级
Max_iteration = 50 # 迭代次数

def objective_function(params):

    x1 = params[0]
    x2 = params[1]
    k1 = int(round(x1))


    print("k1=", params[0])
    print("x2=", params[1])


    clf = AdaBoostClassifier(random_state=41,n_estimators= k1,learning_rate = 0.1)
    clf.fit(X_train, y_train)

    # 预测测试集
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    scores = (1 - accuracy) * 100

    print(scores)
    return np.mean(scores) # 返回平均误差率

# 定义参数的搜索范围
lb = [1,0.1]  # 参数的下界
ub = [100,1] # 参数的上界
''' ------------------------ 获取测试函数细节 F1~F23 ----------------------------------'''

dim = 2

''' ------------------------ 开普勒算法求解 ----------------------------------'''
x = KOA.KOA(objective_function, lb, ub, dim, SearchAgents_no, Max_iteration)

''' ------------------------ 求解结果 ----------------------------------'''
IterCurve = x.convergence
Best_fitness = x.best
Best_Pos = x.bestIndividual

''' ------------------------ 绘图 ----------------------------------'''
func_description = "My Objective Function"
part1 = ['KOA', func_description]
name1 = ' '.join(part1)
plt.figure(1)
plt.plot(IterCurve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title(name1, fontsize='large')
label = [name1]
plt.legend(label, loc='upper right')
plt.savefig('./KOA_Python.jpg')
plt.show()

print(Best_Pos)

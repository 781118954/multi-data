'''
% ----------------------- README ------------------------------------------
%   Author and programmer: Reda Mohamed & Mohamed Abdel-Basset
%   Main paper:
%               Kepler optimization algorithm: A new metaheuristic algorithm
%               inspired by Kepler laws of planetary motion
% -------------- 最后一次修改：2023/1/3 -----------------------------------
% -------------------  欢迎关注₍^.^₎♡  ------------------------------------
% -------------- 项目：Kepler optimization algorithm (KOA) ----------------
% -------------- 微信公众号：KAU的云实验台(可咨询定制) --------------------
% -------------- CSDN：KAU的云实验台 --------------------------------------
% -------------- 付费代码(更全)：https://mbd.pub/o/author-a2iWlGtpZA== ----
% -------------- 免费代码：公众号后台回复"资源" ---------------------------
% -------------------------------------------------------------------------
'''
import random
import numpy as np
import math
from solution import solution
import time


def SortFitness(Fit):
    '''适应度排序'''
    '''
    输入为适应度值
    输出为排序后的适应度值，和索引
    '''
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index

def Bounds(s, Lb, Ub):
    """
    越界随机初始化
    :param s: 编码
    :param Lb: 上街
    :param Ub: 下届
    :return: 新个体
    """
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[i]:
            temp[i] = Lb[i]+(Ub[i]-Lb[i])*np.random.rand(1, 1)
        elif temp[i] > Ub[i]:
            temp[i] = Lb[i]+(Ub[i]-Lb[i])*np.random.rand(1, 1)

    return temp


def KOA(objf, lb, ub, dim, SearchAgents_no, Max_iter):


    '''  定义  '''
    Sun_Pos = np.zeros(dim) # 太阳
    Sun_Score = float("inf")
    # 初始化收敛曲线
    convergence_curve = np.zeros(Max_iter)

    '''  控制参数  '''
    Tc = 3 # a2中的参数
    M0 = 0.1 # 引力中u的计算参数
    lambda1 = 15

    # 判断是否为向量
    if not isinstance(lb, list):
        # 向量化
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    ''' 初始化  '''
    # 初始化种群
    X = np.asarray(
        [x * (ub - lb) + lb for x in np.random.uniform(0, 1, (SearchAgents_no, dim))]
    )
    # 初始化偏心率和轨道周期
    orbital = np.random.random(SearchAgents_no)
    T = abs(np.random.randn(SearchAgents_no))

    fitness = np.zeros(SearchAgents_no)

    # 更新适应度
    for i in range(0, SearchAgents_no):
        # 适应度
        fitness[i] = objf(X[i, :])
        # 更新猎物位置
        if fitness[i] < Sun_Score:
            Sun_Score = fitness[i]
            Sun_Pos = X[i, :].copy()

    # 保存结果
    s = solution()
    R = np.zeros(SearchAgents_no)
    MS = np.zeros(SearchAgents_no)
    m = np.zeros(SearchAgents_no)
    Rnorm = np.zeros(SearchAgents_no)
    MSnorm = np.zeros(SearchAgents_no)
    Mnorm = np.zeros(SearchAgents_no)
    Fg = np.zeros(SearchAgents_no)
    a1 = np.zeros(SearchAgents_no)
    V = np.zeros([SearchAgents_no,dim])

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    t = 0  # Loop counter

    # 迭代
    while t < Max_iter:
        # 对适应度值排序
        fitness, sortIndex = SortFitness(fitness)
        worstFitness = fitness[-1]
        M=M0*(math.exp(-lambda1*(t/Max_iter)))

        # 行星与太阳的欧氏距离
        for i in range(SearchAgents_no):
            for j in range(dim):
                R[i] = R[i] + (Sun_Pos[j]-X[i,j])**2
            R[i] = math.sqrt(R[i])

        # 太阳与行星质量
        for i in range(SearchAgents_no):
            sum = 0
            for k in range(SearchAgents_no):
                sum = sum + (fitness[k] - worstFitness)
            MS[i] = random.random() * (Sun_Score - worstFitness) / sum
            m[i] = (fitness[i] - worstFitness) / sum

        # 引力
        for i in range(SearchAgents_no):
            Rnorm[i] = (R[i] - np.min(R)) / (np.max(R) - np.min(R)) # TheR(Eq.(24))
            MSnorm[i] = (MS[i] - np.min(MS)) / (np.max(MS) - np.min(MS)) # The normalized MS
            Mnorm[i] = (m[i] - np.min(m)) / (np.max(m) - np.min(m)) # The normalized m
            Fg[i] = orbital[i] * M * ((MSnorm[i] * Mnorm[i]) / (Rnorm[i] * Rnorm[i] + 10E-8)) + random.random()# Eq.(6)
            # a1 表示椭圆轨道的半长轴
            a1[i] = random.random()*(T[i]**2*(M*(MS[i]+m[i])/(4*math.pi*math.pi)))**(1/3) # Eq.(23)

        for i in range(0, SearchAgents_no):
            a2 = -1 + -1 * (t % (Max_iter / Tc)) / (Max_iter / Tc)
            n = (a2 - 1) *  random.random() + 1
            a = random.randint(0,SearchAgents_no-1)# An index of a solution selected at random
            b = random.randint(0, SearchAgents_no-1)
            sd = np.random.random(dim)
            ss = random.random()
            U1 = sd < ss
            U11 = np.zeros(dim)
            for iii in range(dim):
                if U1[iii]:
                    U11[iii] = 1
                else:
                    U11[iii] = 0
            U1 = U11
            O_P = X[i,:]
            if random.random() < random.random():
                # Step 6: Updating distance with the Sun 更新和太阳的距离
                # h is an adaptive factor for controlling the distance between the Sun and the current planet at time t
                h = (1 / (math.exp(n * np.random.randn())))
                #  An verage vector based on three solutions: the Current solution, best-so-far solution, and randomly-selected solution
                Xm = (X[b,:] + Sun_Pos + X[i,:]) / 3.0
                X[i,:]=X[i,:]*U1 + (Xm + h * (Xm - X[a,:]))*(1 - U1)
            else:
                if random.random() < 0.5: # Eq.(18)
                    f = 1
                else:
                    f = -1
                L = (M * (MS[i] + m[i]) * abs((2 / (R[i] + 10E-8)) - (1 / (a1[i] + 10E-8)))) ** (0.5)
                U = np.zeros(dim)
                for ii in range(dim):
                    ff = sd[ii] > random.random()
                    if ff:
                        U[ii] = 1
                    else:
                        U[ii] = 0

                if Rnorm[i] < 0.5: # Eq.(13)
                    M = (random.random() * (1 - ss) + ss) # Eq.(16)
                    l = L * M * U #  Eq.(14)
                    Mv = (random.random() * (1 - sd) + sd)
                    l1 = L * Mv * (1 - U)
                    V[i,:] = l * (2 * random.random() * X[i,:] - X[a,:])+l1 * (X[b,:] - X[a,:])+(
                                1 - Rnorm[i]) * f * U1 * np.random.random(dim) * (ub - lb) # Eq.(13a)
                else:
                    U2 = random.random() > random.random()
                    if U2:
                        U2 = 1
                    else:
                        U2 = 0
                    V[i,:]= random.random() * L * (X[a,:] - X[i,:])+(1 - Rnorm[i]) * f * U2 * np.random.random(dim) * (random.random()*ub - lb) #Eq.(13b)

                # Step 4: Escaping from the local optimum 逃出局部最优
                # % Update the flag f to opposite or leave the search direction of the current planet
                if  random.random() < 0.5: # Eq.(18)
                    f = 1
                else:
                    f = -1
                # Step 5 更新天体位置
                X[i,:]=((X[i,:] + V[i,:] * f)+(Fg[i] + abs(np.random.randn())) * U * (Sun_Pos - X[i,:])) #  Eq.(25)

                # 精英
                Xnew = Bounds(X[i, :], lb, ub)
                Fnew = objf(Xnew)
                if Fnew<fitness[i]:
                    fitness[i] = Fnew
                    X[i,:] = Xnew
                    if fitness[i]<Sun_Score:
                        Sun_Score = fitness[i]
                        Sun_Pos = X[i,:]
                convergence_curve[t] = Sun_Score

        t = t + 1
        if t % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(t)
                    + " the best fitness is "
                    + str(Sun_Score)
                ]
            )


    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "KOA"
    s.objfname = objf.__name__
    s.best = Sun_Score
    s.bestIndividual = Sun_Pos

    return s


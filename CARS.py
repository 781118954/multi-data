import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


import warnings
warnings.filterwarnings("ignore")

# 读取数据
data_path = r"yumi.csv"  # 数据
df = pd.read_csv('yumi.csv')
feature_names = df.columns.tolist()
data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=1)

# 划分数据和标签
cols = data.shape[1]
data_x = data[:, :-1]
y = data[:, cols - 1:cols]
y = np.array(y.ravel())  # 数列平铺
y = np.squeeze(y)
print("data_x：", data_x.shape)

X = data_x

def PC_Cross_Validation(X, y, pc=20, cv=10):
    '''
        x :光谱矩阵 nxm
        y :浓度阵 （化学值）
        pc:最大主成分数
        cv:交叉验证数量
    return :
        RMSECV:各主成分数对应的RMSECV
        PRESS :各主成分数对应的PRESS
        rindex:最佳主成分数
    '''
    kf = KFold(n_splits=cv)
    RMSECV = []
    for i in range(pc):
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex

def Cross_Validation(X, y, pc=20, cv=10):
    '''
     x :光谱矩阵 nxm
     y :浓度阵 （化学值）
     pc:最大主成分数
     cv:交叉验证数量
     return :
     RMSECV:各主成分数对应的RMSECV
    '''
    kf = KFold(n_splits=cv)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean

def CARS_Cloud(X, y, N=20, f=20  , cv=10):
    p = 0.8
    m, n = X.shape
    u = np.power((n / 2), (1 / (N - 1)))
    k = (1 / (N - 1)) * np.log(n / 2)
    cal_num = np.round(m * p)
    # val_num = m - cal_num
    b2 = np.arange(n)
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))
    WaveData = []
    # Coeff = []
    WaveNum = []
    RMSECV = []
    r = []
    for i in range(1, N + 1):
        r.append(u * np.exp(-1 * k * i))
        wave_num = int(np.round(r[i - 1] * n))
        WaveNum = np.hstack((WaveNum, wave_num))
        cal_index = np.random.choice \
            (np.arange(m), size=int(cal_num), replace=False)
        wave_index = b2[:wave_num].reshape(1, -1)[0]
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        # xcal = xcal[:,wave_index].reshape(-1,wave_num)
        ycal = y[cal_index]
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1, -1)
        wnum = n - wave_num
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData) == 0:
            WaveData = d
        else:
            WaveData = np.vstack((WaveData, d.reshape(1, -1)))

        if wave_num < f:
            f = wave_num

        pls = PLSRegression(n_components=f)
        pls.fit(xcal, ycal)
        beta = pls.coef_
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=0)
        coef = copy.deepcopy(beta)
        coeff = coef[b2, :].reshape(len(b2), -1)
        # cb = coeff[:wave_num]
        #
        # if wnum > 0:
        #     cb = np.vstack((cb, np.full((wnum, 1), -1)))
        # if len(Coeff) == 0:
        #     Coeff = copy.deepcopy(cb)
        # else:
        #     Coeff = np.hstack((Coeff, cb))
        rmsecv, rindex = PC_Cross_Validation(xcal, ycal, f, cv)
        RMSECV.append(Cross_Validation(xcal, ycal, rindex + 1, cv))
    # CoeffData = Coeff.T

    WAVE = []
    # COEFF = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        # cd = CoeffData[i, :]
        WD = np.ones((len(wd)))
        # CO = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
                # CO[j] = 0
            else:
                WD[j] = wd[ind[0]]
                # CO[j] = cd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))
        # if len(COEFF) == 0:
        #     COEFF = copy.deepcopy(CO)
        # else:
        #     COEFF = np.vstack((WAVE, CO.reshape(1, -1)))

    MinIndex = np.argmin(RMSECV)
    Optimal = WAVE[MinIndex, :]
    boindex = np.where(Optimal != 0)
    OptWave = boindex[0]

    fig = plt.figure(figsize=(8, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 20
    plt.subplot(311)
    plt.xlabel('Number of iterations', fontsize=fonts)
    plt.ylabel('Features', fontsize=fonts)
    plt.title('best iteration count：' + str(MinIndex), fontsize=fonts)
    plt.plot(np.arange(N), WaveNum)
    plt.xticks(range(0, N, 2))
    plt.tick_params(axis='x', which='major', labelsize=20)
    plt.tick_params(axis='y', which='major', labelsize=20)

    plt.subplot(313)
    plt.xlabel('Number of iterations', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    plt.plot(np.arange(N), RMSECV)
    plt.xticks(range(0, N, 2))
    plt.tick_params(axis='x', which='major', labelsize=20)
    plt.tick_params(axis='y', which='major', labelsize=20)

    # plt.subplot(313)
    # plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    # plt.ylabel('各变量系数值', fontsize=fonts)
    # plt.plot(COEFF)
    # plt.vlines(MinIndex, -1e3, 1e3, colors='r')
    plt.savefig('cars_laman_流程图.png')
    plt.show()
    return OptWave


for seed in range(40, 41):
    np.random.seed(seed)  # 设置随机种子
    print(f"Running with seed: {seed}")

    # 这里调用您的CARS_Cloud函数，传入X和y
    lis = CARS_Cloud(X, y)
    print("获取波段数：", len(lis))
    print(lis)
    # 根据波段索引选择数据
    X_ = X[:, lis]
    print(X_)

    # 转换为 pandas DataFrame
    data_with_labels = np.column_stack([X_, y])
    df_normalized = pd.DataFrame(data_with_labels)

    header_row = pd.DataFrame({f'Index': list(range(1, len(lis) + 1))})

    df_with_header = pd.concat([header_row.T, df_normalized], ignore_index=True)
    df_selected_features = df_with_header.drop(df_with_header.index[0])

    # 保存为 CSV 文件，文件名包含种子编号
    file_name = f"feature-extraction_CARS_result{seed}.csv"
    df_selected_features.to_csv(file_name, index=False)
    print(f"特征提取数据已保存为 {file_name}!")

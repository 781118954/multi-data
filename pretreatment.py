#!/usr/bin/env bin
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from scipy.signal import savgol_filter
from copy import deepcopy
import pywt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter


'''
参考博客
https://blog.csdn.net/Joseph__Lagrange/article/details/95302398
https://blog.csdn.net/Joseph__Lagrange/article/details/95302953
'''


class PPPPPretreatment:




    def SG(self, data, w=7, p=3, d=0):
        """
        SG平滑  可以在这里调整参数  w代表窗口 p代表多项式阶数  d代表导数
        多项式的阶数决定了用于拟合的多项式的程度。更高的多项式阶可以 捕获更复杂的趋势 ，但也可能引入不必要的振荡。
        d为0代表不求导  d为1代表一阶导    过程即先平滑  后求导

        """
        data_copy = deepcopy(data)
        if isinstance(data_copy, pd.DataFrame):
            data_copy = data_copy.values

        data_copy = savgol_filter(data_copy, w, polyorder=p, deriv=d)
        return data_copy




    def StandardScaler(self,data):

        scaler = StandardScaler()

        # 使用 StandardScaler 对数据进行标准化
        standardized_data = scaler.fit_transform(data)
        return standardized_data




    def msc(self,input_data, reference=None):
        """
        多元散射校正（MSC）的实现。

        参数:
        input_data: numpy array，形状为 (样本数, 波长数) 的光谱数据。
        reference: 可选，用于校正的参考光谱。如果为 None，则使用输入数据的平均光谱。

        返回:
        校正后的数据。
        """
        # 如果没有提供参考，则使用所有样本的平均光谱

        if reference is None:
            reference = np.mean(input_data, axis=0)

        # 初始化校正后的数据数组
        corrected_data = np.zeros_like(input_data)

        # 对每个样本进行处理
        for i in range(input_data.shape[0]):
            # 获取当前样本
            sample = input_data[i, :]

            # 计算回归系数
            fit = np.polyfit(reference, sample, 1, full=True)

            # 应用校正
            corrected_data[i, :] = (sample - fit[0][1]) / fit[0][0]

        return corrected_data




    def snv(self,input_data):
        # 对每一行应用SNV转换
        snv_transformed = (input_data - input_data.mean(axis=1, keepdims=True)) / input_data.std(axis=1, keepdims=True)
        return snv_transformed


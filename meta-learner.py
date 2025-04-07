import numpy as np
import datetime
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

start_time = datetime.datetime.now()  # 计时

# 最大最小值归一化
def MMS(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MinMaxScaler :(n_samples, n_features)
       """
    return MinMaxScaler().fit_transform(data)
#混淆矩阵
def CM():
    confusion = confusion_matrix(y_test, y_pred)

    # 计算每个类别的准确率
    confusion = confusion / confusion.sum(axis=1)[:,np.newaxis]
    # 创建热力图
    plt.figure(figsize=(8, 6))
    custom_cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(confusion, annot=True, fmt=".2f", cmap=custom_cmap,
                xticklabels=['P 1', 'P 2', 'P 3', 'P 4', 'P 5', 'P 6', 'P 7'],
                yticklabels=['A 1', 'A 2', 'A 3', 'A 4', 'A 5', 'A 6', 'A 7'])  # P表示Predicted预测值，A表示Actual 实际值
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # 生成文件名并保存图像为JPEG格式
    # file_name = f'{self.cm_path}\{name}_{self.current_time}_{i + 1}.jpg'
    # plt.savefig(file_name, format='jpg', dpi=300)  # 使用dpi参数设置图像分辨率（可选）
    plt.savefig(f'hunxiao + {clf}.jpg')
    plt.show()


# 划分数据和标签
data_path = 'mult.csv'  # 数据
data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=1)

data_x = data[:, :-1]
print("data_x：", data_x .shape)
# datareprocessing_path = 'C:/Users/86138/Desktop/dataMSC.csv'  #波长
# Data_WAVE = wave(data_x)  #改这里的函数名就可以得到不同的预处理
#Data_MMS = MMS(data_x)
Data_MMS = data_x

# 划分数据标签
cols = data.shape[1]
X = Data_MMS
y = data[:, cols - 1:cols]
y = np.array(y.ravel())  # 数列平铺
y = np.squeeze(y)
print("X_filtered", X.shape)

seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 定义KNN模型，其K值由x传入
clf1 = KNeighborsClassifier(n_neighbors=76, p=3, weights='distance')
clf2 = SVC(C=828.70, gamma=0.33, probability=True)
clf3 = DecisionTreeClassifier(max_depth=687, max_features=0.518)
clf4 = MLPClassifier(solver='sgd', learning_rate_init=0.1,random_state=42, max_iter=1000)
clf5 = RandomForestClassifier(n_estimators=100, random_state=42)
clf6 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5],
                            use_probas=True,
                            # average_probas=False,
                            meta_classifier=lr,
                            random_state=seed)

for clf in (clf1, clf2, clf3, clf4, clf5, clf6, sclf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pro = clf.predict_proba(X_test)
    #print(clf.__class__.__name__, 'Accuracy:', accuracy_score(y_test, y_pred))
    #print(clf.__class__.__name__, 'Precison:', precision_score(y_test, y_pred, average='macro'))
    #print(clf.__class__.__name__, 'Recall  :', recall_score(y_test, y_pred, average='macro'))
    #print(clf.__class__.__name__, 'F1      :', f1_score(y_test, y_pred, average='macro'))
    #print(classification_report(y_test, y_pred))
    CM()

    n_classes = len(np.unique(y))
    y_test_binarized = label_binarize(y_test, classes=list(range(n_classes)))

# 计算每个类别的ROC曲线和AUC值
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pro[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(f'roc + {clf}.jpg')
    plt.show()
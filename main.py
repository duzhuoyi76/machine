import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

# 加载数据集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('testA.csv')

# 数据预处理
X = np.stack(train_data['heartbeat_signals'].apply(lambda x: np.array(x.split(','), dtype=np.float32)).values)
y = pd.get_dummies(train_data['label']).values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train) 
# 预测
y_pred_raw = clf.predict_proba(X_test)
y_pred = np.array([pred[:, 1] for pred in y_pred_raw]).T

# 计算abs-sum
abs_sum = np.mean(np.abs(y_test - y_pred))
print('Abs-sum:', abs_sum)

# 预测测试集并保存结果
test_X = np.stack(test_data['heartbeat_signals'].apply(lambda x: np.array(x.split(','), dtype=np.float32)).values)
test_pred_raw = clf.predict_proba(test_X)
test_pred = np.array([pred[:, 1] for pred in test_pred_raw]).T
result = pd.DataFrame(test_pred, columns=['label_0', 'label_1', 'label_2', 'label_3'])
result['id'] = test_data['id']
result.to_csv('submission.csv', index=False)
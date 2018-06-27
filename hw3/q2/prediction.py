import time
import calendar
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from functools import reduce
from collections import defaultdict

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score


# User-Product Pairs
pairs = raw_data[np.logical_and(
    raw_data['month'] >= 5,
    raw_data['month'] <= 8)].groupby(['vipno', 'pluno']).agg({'no': 'count'})
pairs = pairs.drop(columns=['no'])
pairs = pairs.reset_index()

train_start_month = 2
train_end_month = 5
test_start_month = 4
test_end_month = 6
predict_start_month = 5
predict_end_month = 7

data = []
for index, delta_month in enumerate(range(month_window_length)):
    window_start_month = start_month + delta_month
    window_end_month = start_month + delta_month + k - 1
    
    month_pairs = pairs.copy()
    month_pairs['start_month'] = window_start_month
    month_pairs['end_month'] = window_end_month
    
    next_month_data = raw_data.query(f'month == {window_end_month + 1}')[['vipno', 'pluno', 'no']] \
        .drop_duplicates(['vipno', 'pluno'])
    month_pairs = pd.merge(month_pairs, next_month_data, on=['vipno', 'pluno'], how='left')
    month_pairs = month_pairs.rename(columns={'no': 'y'})
    month_pairs['y'] = month_pairs['y'].notnull().astype('int')
    
    month_pairs = pd.merge(month_pairs, feature_database['user_product'],
                           on=['vipno', 'pluno', 'start_month', 'end_month'], how='left')
    month_pairs = pd.merge(month_pairs, feature_database['user'],
                           on=['vipno', 'start_month', 'end_month'], how='left')
    month_pairs = pd.merge(month_pairs, feature_database['product'],
                           on=['pluno', 'start_month', 'end_month'], how='left')
    
    data.append(month_pairs)

data = pd.concat(data).reset_index().drop(columns=['index'])

train_data = data.query(f'{train_start_month} <= start_month <= {train_end_month - k + 1}')
test_data = data.query(f'{test_start_month} <= start_month <= {test_end_month - k + 1}')
to_predict_data = data.query(f'{predict_start_month} <= start_month <= {predict_end_month - k + 1}')
train_X = train_data.drop(columns=['vipno', 'pluno', 'start_month', 'end_month', 'y'])
train_y = train_data['y']
test_X = test_data.drop(columns=['vipno', 'pluno', 'start_month', 'end_month', 'y'])
test_y = test_data['y']
to_predict_X = to_predict_data.drop(columns=['vipno', 'pluno', 'start_month', 'end_month', 'y'])

pd.options.mode.chained_assignment = None

models = {
    'K Neighbors': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier()
}

for name, clf in models.items():
    clf_name = clf.__class__.__name__
    
    now = time.time()
    clf.fit(train_X, train_y)
    predict_y = clf.predict(test_X)
    time_elapsed = time.time() - now
    
    precision = precision_score(test_y, predict_y)
    recall = recall_score(test_y, predict_y)
    f1 = f1_score(test_y, predict_y)
    
    print(f'{name} ({clf_name})')
    print(f'precision: {precision * 100:.2f}% '
          f'recall: {recall * 100:.2f}% '
          f'F: {f1 * 100:.2f}% '
          f'({time_elapsed:.2f}ms)')
    
    predict_y = clf.predict(to_predict_X)
    to_predict_data['y'] = predict_y
    to_predict_data['y'] = to_predict_data['y'].apply(lambda x: 'Yes' if x > 0 else 'No')
    to_predict_data[['vipno', 'pluno', 'y']].to_csv(f'1452669_2b_{clf_name}.txt', header=False, index=False)



# User Purchase
pairs = raw_data[np.logical_and(
    raw_data['month'] >= 5,
    raw_data['month'] <= 8)].groupby(['vipno']).agg({'no': 'count'})
pairs = pairs.drop(columns=['no'])
pairs = pairs.reset_index()

data = []
for index, delta_month in enumerate(range(month_window_length)):
    window_start_month = start_month + delta_month
    window_end_month = start_month + delta_month + k - 1
    
    month_pairs = pairs.copy()
    month_pairs['start_month'] = window_start_month
    month_pairs['end_month'] = window_end_month
    
    next_month_data = raw_data.query(f'month == {window_end_month + 1}')[['vipno', 'no']] \
        .drop_duplicates(['vipno'])
    month_pairs = pd.merge(month_pairs, next_month_data, on=['vipno'], how='left')
    month_pairs = month_pairs.rename(columns={'no': 'y'})
    month_pairs['y'] = month_pairs['y'].notnull().astype('int')
    
    month_pairs = pd.merge(month_pairs, feature_database['user'],
                           on=['vipno', 'start_month', 'end_month'], how='left')
    
    data.append(month_pairs)

data = pd.concat(data).reset_index().drop(columns=['index'])

train_data = data.query(f'{train_start_month} <= start_month <= {train_end_month - k + 1}')
test_data = data.query(f'{test_start_month} <= start_month <= {test_end_month - k + 1}')
to_predict_data = data.query(f'{predict_start_month} <= start_month <= {predict_end_month - k + 1}')
train_X = train_data.drop(columns=['vipno', 'start_month', 'end_month', 'y'])
train_y = train_data['y']
test_X = test_data.drop(columns=['vipno', 'start_month', 'end_month', 'y'])
test_y = test_data['y']
to_predict_X = to_predict_data.drop(columns=['vipno', 'start_month', 'end_month', 'y'])

for name, clf in models.items():
    clf_name = clf.__class__.__name__
    
    now = time.time()
    clf.fit(train_X, train_y)
    predict_y = clf.predict(test_X)
    time_elapsed = time.time() - now
    
    precision = precision_score(test_y, predict_y)
    recall = recall_score(test_y, predict_y)
    f1 = f1_score(test_y, predict_y)
    
    print(f'{name} ({clf_name})')
    print(f'precision: {precision * 100:.2f}% '
          f'recall: {recall * 100:.2f}% '
          f'F: {f1 * 100:.2f}% '
          f'({time_elapsed:.2f}ms)')
    
    predict_y = clf.predict(to_predict_X)
    to_predict_data['y'] = predict_y
    to_predict_data['y'] = to_predict_data['y'].apply(lambda x: 'Yes' if x > 0 else 'No')
    to_predict_data[['vipno', 'y']].to_csv(f'1452669_2ci_{clf_name}.txt', header=False, index=False)


# User-Brand Pairs
pairs = raw_data[np.logical_and(
    raw_data['month'] >= 5,
    raw_data['month'] <= 8)].groupby(['vipno', 'bndno']).agg({'no': 'count'})
pairs = pairs.drop(columns=['no'])
pairs = pairs.reset_index()
data = []
for index, delta_month in enumerate(range(month_window_length)):
    window_start_month = start_month + delta_month
    window_end_month = start_month + delta_month + k - 1
    
    month_pairs = pairs.copy()
    month_pairs['start_month'] = window_start_month
    month_pairs['end_month'] = window_end_month
    
    next_month_data = raw_data.query(f'month == {window_end_month + 1}')[['vipno', 'bndno', 'no']] \
        .drop_duplicates(['vipno', 'bndno'])
    month_pairs = pd.merge(month_pairs, next_month_data, on=['vipno', 'bndno'], how='left')
    month_pairs = month_pairs.rename(columns={'no': 'y'})
    month_pairs['y'] = month_pairs['y'].notnull().astype('int')
    
    month_pairs = pd.merge(month_pairs, feature_database['user_brand'],
                           on=['vipno', 'bndno', 'start_month', 'end_month'], how='left')
    month_pairs = pd.merge(month_pairs, feature_database['user'],
                           on=['vipno', 'start_month', 'end_month'], how='left')
    month_pairs = pd.merge(month_pairs, feature_database['brand'],
                           on=['bndno', 'start_month', 'end_month'], how='left')
    
    data.append(month_pairs)

data = pd.concat(data).reset_index().drop(columns=['index'])

train_data = data.query(f'{train_start_month} <= start_month <= {train_end_month - k + 1}')
test_data = data.query(f'{test_start_month} <= start_month <= {test_end_month - k + 1}')
to_predict_data = data.query(f'{predict_start_month} <= start_month <= {predict_end_month - k + 1}')
train_X = train_data.drop(columns=['vipno', 'bndno', 'start_month', 'end_month', 'y'])
train_y = train_data['y']
test_X = test_data.drop(columns=['vipno', 'bndno', 'start_month', 'end_month', 'y'])
test_y = test_data['y']
to_predict_X = to_predict_data.drop(columns=['vipno', 'bndno', 'start_month', 'end_month', 'y'])

for name, clf in models.items():
    clf_name = clf.__class__.__name__
    
    now = time.time()
    clf.fit(train_X, train_y)
    predict_y = clf.predict(test_X)
    time_elapsed = time.time() - now
    
    precision = precision_score(test_y, predict_y)
    recall = recall_score(test_y, predict_y)
    f1 = f1_score(test_y, predict_y)
    
    print(f'{name} ({clf_name})')
    print(f'precision: {precision * 100:.2f}% '
          f'recall: {recall * 100:.2f}% '
          f'F: {f1 * 100:.2f}% '
          f'({time_elapsed:.2f}ms)')
    
    predict_y = clf.predict(to_predict_X)
    to_predict_data['y'] = predict_y
    to_predict_data['y'] = to_predict_data['y'].apply(lambda x: 'Yes' if x > 0 else 'No')
    to_predict_data[['vipno', 'bndno', 'y']].to_csv(f'1452669_2cii_{clf_name}.txt', header=False, index=False)



# User-Category Pairs
pairs = raw_data[np.logical_and(
    raw_data['month'] >= 5,
    raw_data['month'] <= 8)].groupby(['vipno', 'dptno']).agg({'no': 'count'})
pairs = pairs.drop(columns=['no'])
pairs = pairs.reset_index()

data = []
for index, delta_month in enumerate(range(month_window_length)):
    window_start_month = start_month + delta_month
    window_end_month = start_month + delta_month + k - 1
    
    month_pairs = pairs.copy()
    month_pairs['start_month'] = window_start_month
    month_pairs['end_month'] = window_end_month
    
    next_month_data = raw_data.query(f'month == {window_end_month + 1}')[['vipno', 'dptno', 'no']] \
        .drop_duplicates(['vipno', 'dptno'])
    month_pairs = pd.merge(month_pairs, next_month_data, on=['vipno', 'dptno'], how='left')
    month_pairs = month_pairs.rename(columns={'no': 'y'})
    month_pairs['y'] = month_pairs['y'].notnull().astype('int')
    
    month_pairs = pd.merge(month_pairs, feature_database['user_category'],
                           on=['vipno', 'dptno', 'start_month', 'end_month'], how='left')
    month_pairs = pd.merge(month_pairs, feature_database['user'],
                           on=['vipno', 'start_month', 'end_month'], how='left')
    month_pairs = pd.merge(month_pairs, feature_database['category'],
                           on=['dptno', 'start_month', 'end_month'], how='left')
    
    data.append(month_pairs)

data = pd.concat(data).reset_index().drop(columns=['index'])

train_data = data.query(f'{train_start_month} <= start_month <= {train_end_month - k + 1}')
test_data = data.query(f'{test_start_month} <= start_month <= {test_end_month - k + 1}')
to_predict_data = data.query(f'{predict_start_month} <= start_month <= {predict_end_month - k + 1}')
train_X = train_data.drop(columns=['vipno', 'dptno', 'start_month', 'end_month', 'y'])
train_y = train_data['y']
test_X = test_data.drop(columns=['vipno', 'dptno', 'start_month', 'end_month', 'y'])
test_y = test_data['y']
to_predict_X = to_predict_data.drop(columns=['vipno', 'dptno', 'start_month', 'end_month', 'y'])

for name, clf in models.items():
    clf_name = clf.__class__.__name__
    
    now = time.time()
    clf.fit(train_X, train_y)
    predict_y = clf.predict(test_X)
    time_elapsed = time.time() - now
    
    precision = precision_score(test_y, predict_y)
    recall = recall_score(test_y, predict_y)
    f1 = f1_score(test_y, predict_y)
    
    print(f'{name} ({clf_name})')
    print(f'precision: {precision * 100:.2f}% '
          f'recall: {recall * 100:.2f}% '
          f'F: {f1 * 100:.2f}% '
          f'({time_elapsed:.2f}ms)')
    
    predict_y = clf.predict(to_predict_X)
    to_predict_data['y'] = predict_y
    to_predict_data['y'] = to_predict_data['y'].apply(lambda x: 'Yes' if x > 0 else 'No')
    to_predict_data[['vipno', 'dptno', 'y']].to_csv(f'1452669_2ciii_{clf_name}.txt', header=False, index=False)


# User Purchase Amount
pairs = raw_data[np.logical_and(
    raw_data['month'] >= 5,
    raw_data['month'] <= 8)].groupby(['vipno']).agg({'amt': 'sum'})
pairs = pairs.reset_index()

data = []
for index, delta_month in enumerate(range(month_window_length)):
    window_start_month = start_month + delta_month
    window_end_month = start_month + delta_month + k - 1
    
    month_pairs = pairs.copy()
    month_pairs['start_month'] = window_start_month
    month_pairs['end_month'] = window_end_month
    
    next_month_data = raw_data.query(f'month == {window_end_month + 1}').groupby(['vipno']).agg({'amt': 'sum'})
    month_pairs = pd.merge(month_pairs, next_month_data, on=['vipno'], how='left')
    month_pairs = pd.merge(month_pairs, feature_database['user'],
                           on=['vipno', 'start_month', 'end_month'], how='left')
    
    data.append(month_pairs)

data = pd.concat(data).reset_index().drop(columns=['index']).fillna(0)

train_data = data.query(f'{train_start_month} <= start_month <= {train_end_month - k + 1}')
test_data = data.query(f'{test_start_month} <= start_month <= {test_end_month - k + 1}')
to_predict_data = data.query(f'{predict_start_month} <= start_month <= {predict_end_month - k + 1}')
train_X = train_data.drop(columns=['vipno', 'start_month', 'end_month', 'amt'])
train_y = train_data['amt']
test_X = test_data.drop(columns=['vipno', 'start_month', 'end_month', 'amt'])
test_y = test_data['amt']
to_predict_X = to_predict_data.drop(columns=['vipno', 'start_month', 'end_month', 'amt'])




# overall
models = {
    'Gaussian Naive Bayes': BayesianRidge(),
    'K Neighbors': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Bagging': BaggingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor()
}
markers = '.^o+*Dpsx'
down_sampling_count = 10
plt.style.use('ggplot')

for (name, clf), marker in zip(models.items(), markers):
    clf_name = clf.__class__.__name__
    
    now = time.time()
    clf.fit(train_X, train_y)
    predict_y = clf.predict(test_X)
    time_elapsed = time.time() - now
    
    predict_y_error = np.abs(predict_y - test_y)
    predict_y_error = predict_y_error.sort_values()
    down_samples = np.linspace(0, predict_y.size - 1, down_sampling_count, dtype=int)
    predict_y_sampled = np.asarray(predict_y_error.take(down_samples))
    
    xticks = np.arange(0, 1, 1 / down_sampling_count)
    plt.plot(xticks, predict_y_sampled,
             linestyle='--', marker=marker, alpha=0.8)
    plt.xticks(xticks)
    
    print(f'{name} ({clf_name})')
    print(f'Avg Error: {calculate_area(predict_y_sampled):.2f} '
          f'({time_elapsed:.2f}ms)')
    
    predict_y = clf.predict(to_predict_X)
    to_predict_data['amt'] = predict_y
    to_predict_data[['vipno', 'amt']].to_csv(
        f'1452669_2civ_{clf_name}.txt', header=False, index=False)

plt.title('Comparison of Regressors')
plt.xlabel('CDF')
plt.ylabel('Error (RMB)')
plt.legend(models.keys())
plt.savefig('regressors_comparison.png')
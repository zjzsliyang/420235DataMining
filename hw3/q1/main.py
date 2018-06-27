import os
import math
import time
import numpy
import pickle
import pandas
import logging
from functools import reduce
from datetime import datetime
from operator import itemgetter
from haversine import haversine
import matplotlib.pyplot as plt


import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score

longitude_range = (121.2012049, 121.2183295)
latitude_range = (31.28175691, 31.29339344)
raster_length = 20
data_df_path = os.getcwd() + '/data_df.pkl'
station_df_path = os.getcwd() + '/station_df.pkl'


def preprocessing():
    start = datetime.now()

    data_types = {'IMSI': int, 'MRTime': int}
    for i in range(1, 8):
        data_types[f'RSSI_{i}'] = int
        data_types[f'RNCID_{i}'] = int
        data_types[f'CellID_{i}'] = int
        data_types[f'AsuLevel_{i}'] = int
        data_types[f'SignalLevel_{i}'] = int
    station_types = {'RNCID': int, 'CellID': int}

    data_df = pandas.read_csv('../../data/trajectory_data/2g_data.csv').fillna(-110)
    station_df = pandas.read_csv('../../data/trajectory_data/2g_gongcan.csv')

    for key, value in data_types.items():
        data_df[key] = data_df[key].astype(int)
    for key, value in station_types.items():
        station_df[key] = station_df[key].astype(int)

    data_df_f = open(data_df_path, 'wb')
    pickle.dump(data_df, data_df_f)
    station_df_f = open(station_df_path, 'wb')
    pickle.dump(station_df, station_df_f)
    logging.info('time cost in preprocessing: ' + str(datetime.now() - start))


def rasterize():
    data_df_f = open(data_df_path, 'rb')
    data_df = pickle.load(data_df_f)

    x_dist = haversine((latitude_range[0], longitude_range[0]), (latitude_range[0], longitude_range[1]))
    y_dist = haversine((latitude_range[0], longitude_range[0]), (latitude_range[1], longitude_range[0]))

    x_raster_cnt = int(numpy.ceil(x_dist * 1000 / raster_length))
    y_raster_cnt = int(numpy.ceil(y_dist * 1000 / raster_length))

    x_stride = (longitude_range[1] - longitude_range[0]) / x_raster_cnt
    y_stride = (latitude_range[1] - latitude_range[0]) / y_raster_cnt

    logging.info('the count of x_raster is: ' + str(x_raster_cnt))
    logging.info('the count of y_raster is: ' + str(y_raster_cnt))

    def coordinate_to_raster(longitude, latitude):
        x = math.floor((longitude - longitude_range[0]) / x_stride)
        y = math.floor((latitude - latitude_range[0]) / y_stride)
        return int(y * x_raster_cnt + x)

    def raster_to_longitude(raster):
        x = raster % x_raster_cnt
        return longitude_range[0] + (x + 0.5) * x_stride

    def raster_to_latitude(raster):
        y = raster // x_raster_cnt
        return latitude_range[0] + (y + 0.5) * y_stride

    data_df['raster'] = numpy.vectorize(coordinate_to_raster)(data_df['Longitude'], data_df['Latitude'])
    data_df['r_longitude'] = numpy.vectorize(raster_to_longitude)(data_df['raster'])
    data_df['r_latitude'] = numpy.vectorize(raster_to_latitude)(data_df['raster'])
    data_df_f = open(data_df_path, 'wb')
    pickle.dump(data_df, data_df_f)

    def draw_reference_lines():
        x = longitude_range[0]
        for i in range(x_raster_cnt):
            x += x_stride
            plt.vlines(x, ymin=latitude_range[0], ymax=latitude_range[1], color='r', linewidth=0.5, alpha=0.4)

        y = latitude_range[0]
        for i in range(y_raster_cnt):
            y += y_stride
            plt.hlines(y, xmin=longitude_range[0], xmax=longitude_range[1], color='r', linewidth=0.5, alpha=0.4)

    plt.style.use('default')
    plt.scatter(data_df['Longitude'], data_df['Latitude'], c='black', s=1, alpha=0.25)
    plt.xlim(longitude_range)
    plt.ylim(latitude_range)
    draw_reference_lines()
    plt.savefig('scatter.pdf')

    plt.style.use('default')
    plt.scatter(data_df['r_longitude'], data_df['r_latitude'], color='black', s=1, alpha=0.25)
    plt.xlim(longitude_range)
    plt.ylim(latitude_range)
    draw_reference_lines()
    plt.savefig('raster_plot.pdf')

    models(x_raster_cnt, y_raster_cnt)


def haversine_vec(a_lat, a_lng, b_lat, b_lng):
    R = 6371  # earth radius in km

    a_lat = numpy.radians(a_lat)
    a_lng = numpy.radians(a_lng)
    b_lat = numpy.radians(b_lat)
    b_lng = numpy.radians(b_lng)

    d_lat = b_lat - a_lat
    d_lng = b_lng - a_lng

    d_lat_sq = numpy.sin(d_lat / 2) ** 2
    d_lng_sq = numpy.sin(d_lng / 2) ** 2

    a = d_lat_sq + numpy.cos(a_lat) * numpy.cos(b_lat) * d_lng_sq
    c = 2 * numpy.arctan2(numpy.sqrt(a), numpy.sqrt(1 - a))

    return R * c  # returns distance between a and b in km


def prepare_dataset_with_station(data_df, station_df):
    X = data_df.drop(columns=['IMSI', 'MRTime', 'Longitude', 'Latitude',
                              'SignalLevel_1', 'SignalLevel_2', 'SignalLevel_3', 'SignalLevel_4',
                              'SignalLevel_5', 'SignalLevel_6', 'SignalLevel_7',
                              'AsuLevel_1', 'AsuLevel_2', 'AsuLevel_3', 'AsuLevel_4',
                              'AsuLevel_5', 'AsuLevel_6', 'AsuLevel_7',
                              'raster', 'raster_latitude', 'raster_longitude'])
    for i in range(1, 8):
        X = pandas.merge(X, station_df, how='left',
                         left_on=[f'RNCID_{i}', f'CellID_{i}'],
                         right_on=['RNCID', 'CellID'])
        X = X.rename(index=str,
                     columns={'Longitude': f'Longitude_{i}',
                              'Latitude': f'Latitude_{i}'})
        X = X.drop(columns=['RNCID', 'CellID', f'RNCID_{i}', f'CellID_{i}'])

    X = X.fillna(-1)
    y = data_df[['Longitude', 'Latitude', 'raster']]
    return X, y


def models(x_raster_cnt, y_raster_cnt):
    data_df_f = open(data_df_path, 'rb')
    data_df = pickle.load(data_df_f)
    station_df_f = open(station_df_path, 'rb')
    station_df = pickle.load(station_df_f)

    raster_count = data_df.groupby(['raster'])['raster'] \
        .count().reset_index(name='count') \
        .sort_values('count', ascending=False)
    top_rasters = raster_count[:50]['raster']

    models = {
        'K Neighbors': KNeighborsClassifier(n_neighbors=2),
        'Gaussian Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=50),
        'Random Forest': RandomForestClassifier(max_features=3, n_estimators=45),
        'Bagging': BaggingClassifier(n_estimators=40),
        'AdaBoost': AdaBoostClassifier(n_estimators=150, learning_rate=0.1),
        'Gradient Boosting': GradientBoostingClassifier(max_depth=10, n_estimators=40),
        'XGBoost': None,
    }
    model_error = {}

    # configuration for XGBoost
    param = {'max_depth': 5, 'eta': 1, 'silent': 1,
             'objective': 'multi:softmax', 'num_class': x_raster_cnt * y_raster_cnt}
    num_round = 100

    markers = ['.', '^', 'o', '+', '*', 'D', 'p', 's', 'x']
    round_count = 10
    down_sampling_count = 10
    plt.style.use('ggplot')

    X, y = prepare_dataset_with_station(data_df)

    for (name, clf), marker in zip(models.items(), markers):
        print('=' * 50)
        print(f'Classifier: {name}')
        print('=' * 50)

        precision = 0
        recall = 0
        f1 = 0
        y_distances = numpy.repeat(0.0, down_sampling_count)

        for i in range(round_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            now = time.time()
            if name == 'XGBoost':
                dtrain = xgb.DMatrix(X_train, label=y_train['raster'])
                dtest = xgb.DMatrix(X_test)
                bst = xgb.train(param, dtrain, num_round)
                y_predict = bst.predict(dtest)

            else:
                clf.fit(X_train, y_train['raster'])
                y_predict = clf.predict(X_test)
            time_elapsed = time.time() - now

            round_precision = 0
            round_recall = 0
            for raster in top_rasters:
                round_precision += precision_score(y_predict == raster, y_test['raster'] == raster)
                round_recall += recall_score(y_predict == raster, y_test['raster'] == raster)

            round_precision /= len(top_rasters)
            round_recall /= len(top_rasters)

            print(f'(Round {i}) '
                  f'P: {round_precision * 100:.2f}%, '
                  f'R: {round_recall * 100:.2f}%, '
                  f'F: {2 * (round_precision * round_recall) / (round_precision + round_recall) * 100:.2f}%'
                  f' ({time_elapsed:.2f}s)')

            precision += round_precision
            recall += round_recall

            y_predict_latitude = numpy.vectorize(raster_to_latitude)(y_predict)
            y_predict_longitude = numpy.vectorize(raster_to_longitude)(y_predict)
            y_distance = haversine_vec(y_predict_latitude, y_predict_longitude,
                                       y_test['Latitude'], y_test['Longitude'])
            y_distance *= 1000  # unit: meters
            y_distance = y_distance.sort_values()
            y_distance = numpy.asarray(y_distance[::int(y_distance.size / down_sampling_count)], dtype=float)

            y_distances += y_distance

        y_distances /= round_count
        model_error[name] = y_distances

        xticks = numpy.arange(0, 1, 1 / down_sampling_count)
        plt.plot(xticks, y_distances,
                 linestyle='--', marker=marker, alpha=0.8)
        plt.xticks(xticks)

        precision /= round_count
        recall /= round_count
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f'(Overall) '
              f'P: {precision * 100:.2f}%, '
              f'R: {recall * 100:.2f}%, '
              f'F: {f1 * 100:.2f}%')

    plt.title('Comparison of Classifiers')
    plt.xlabel('CDF')
    plt.ylabel('Error (meters)')
    plt.legend(models.keys())
    plt.savefig('classifiers_comparison.png')

    for name, y_distances in model_error.items():
        print(f'{name}: {calculate_area(y_distances):.2f}m')

    IMSI = data_df.groupby(['IMSI']).agg({'MRTime': 'count'}).to_dict()['MRTime']
    train_IMSI = {460011670515939.06: 2521, 460016291512176.94: 2550}
    test_IMSI = {460012796993062.0: 1025}

    clf = RandomForestClassifier(max_features=3, n_estimators=45)

    train_data = data_df.loc[
        numpy.logical_or(data_df['IMSI'] == 460011670515939.06,
                      data_df['IMSI'] == 460016291512176.94)]
    test_data = data_df.loc[data_df['IMSI'] == 460012796993062.00]

    X_train, y_train = prepare_dataset_with_station(train_data)
    X_test, y_test = prepare_dataset_with_station(test_data)

    clf.fit(X_train, y_train['raster'])
    y_predict_test = clf.predict(X_test)
    y_predict_train = clf.predict(X_train)

    y_predict_test_latitude = numpy.vectorize(raster_to_latitude)(y_predict_test)
    y_predict_test_longitude = numpy.vectorize(raster_to_longitude)(y_predict_test)

    y_predict_train_latitude = numpy.vectorize(raster_to_latitude)(y_predict_train)
    y_predict_train_longitude = numpy.vectorize(raster_to_longitude)(y_predict_train)

    y_distance_coarse = haversine_vec(y_predict_test_latitude, y_predict_test_longitude,
                                      y_test['Latitude'], y_test['Longitude'])
    y_distance_coarse *= 1000  # unit: meters
    y_distance_coarse = y_distance_coarse.sort_values()
    down_samples = numpy.linspace(0, y_distance_coarse.size - 1, down_sampling_count, dtype=int)
    y_distance_coarse = numpy.asarray(y_distance_coarse.take(down_samples))

    for each_IMSI, _ in IMSI.items():
        train_data.loc[train_data['IMSI'] == each_IMSI, 'order'] = \
            train_data.loc[train_data['IMSI'] == each_IMSI, 'MRTime'].argsort()
        test_data.loc[test_data['IMSI'] == each_IMSI, 'order'] = \
            test_data.loc[test_data['IMSI'] == each_IMSI, 'MRTime'].argsort()

    train_data['order'] = train_data['order'].astype('int')
    test_data['order'] = test_data['order'].astype('int')

    train_data['pred_lon'] = y_predict_train_longitude
    train_data['pred_lat'] = y_predict_train_latitude
    test_data['pred_lon'] = y_predict_test_longitude
    test_data['pred_lat'] = y_predict_test_latitude

    new_features = ['last_lat', 'last_lon', 'next_lat', 'next_lon',
                    'last_distance', 'next_distance',
                    'last_time_gap', 'next_time_gap',
                    'last_speed', 'next_speed']
    for feature in new_features:
        train_data[feature] = 0
        test_data[feature] = 0

    for data, data_IMSI in zip([train_data, test_data], [train_IMSI, test_IMSI]):
        for each_IMSI, mrtime_count in data_IMSI.items():
            for t in range(mrtime_count):
                last_pos = data.loc[
                    numpy.logical_and(
                        data['IMSI'] == each_IMSI,
                        data['order'] == (t - 1 if t > 0 else 0)),
                    ['pred_lon', 'pred_lat', 'MRTime']].to_dict()
                next_pos = data.loc[
                    numpy.logical_and(
                        data['IMSI'] == each_IMSI,
                        data['order'] == (t + 1 if t < mrtime_count - 1 else mrtime_count - 1)),
                    ['pred_lon', 'pred_lat', 'MRTime']].to_dict()
                this_pos = data.loc[
                    numpy.logical_and(
                        data['IMSI'] == each_IMSI,
                        data['order'] == t),
                    ['pred_lon', 'pred_lat', 'MRTime']].to_dict()

                last_lat = list(last_pos['pred_lat'].values())[0]
                last_lon = list(last_pos['pred_lon'].values())[0]
                last_time = list(last_pos['MRTime'].values())[0]
                next_lat = list(next_pos['pred_lat'].values())[0]
                next_lon = list(next_pos['pred_lon'].values())[0]
                next_time = list(next_pos['MRTime'].values())[0]
                this_lat = list(this_pos['pred_lat'].values())[0]
                this_lon = list(this_pos['pred_lon'].values())[0]
                this_time = list(this_pos['MRTime'].values())[0]

                last_distance = haversine((this_lat, this_lon), (last_lat, last_lon)) * 1000
                next_distance = haversine((this_lat, this_lon), (next_lat, next_lon)) * 1000
                last_time_gap = this_time - last_time
                next_time_gap = next_time - this_time
                last_speed = last_distance / last_time_gap if last_time_gap != 0 else 0
                next_speed = next_distance / next_time_gap if next_time_gap != 0 else 0

                data.loc[numpy.logical_and(
                    data['IMSI'] == each_IMSI,
                    data['order'] == t),
                         ['last_lat', 'last_lon', 'next_lat', 'next_lon']] = \
                    numpy.asarray([last_lat, last_lon, next_lat, next_lon])

                data.loc[numpy.logical_and(
                    data['IMSI'] == each_IMSI,
                    data['order'] == t),
                         ['last_distance', 'next_distance',
                          'last_time_gap', 'next_time_gap',
                          'last_speed', 'next_speed']] = \
                    numpy.asarray([last_distance, next_distance,
                                last_time_gap, next_time_gap,
                                last_speed, next_speed])
    clf = RandomForestRegressor(max_depth=10, n_estimators=100)

    X_train, y_train = prepare_dataset_with_station(train_data)
    X_test, y_test = prepare_dataset_with_station(test_data)

    clf.fit(X_train, y_train['Latitude'])
    y_predict_latitude = clf.predict(X_test)
    clf.fit(X_train, y_train['Longitude'])
    y_predict_longitude = clf.predict(X_test)

    y_distance_fine = haversine_vec(y_predict_latitude, y_predict_longitude,
                                    y_test['Latitude'], y_test['Longitude'])
    y_distance_fine *= 1000  # unit: meters
    y_distance_fine = y_distance_fine.sort_values()
    down_samples = numpy.linspace(0, y_distance_fine.size - 1, down_sampling_count, dtype=int)
    y_distance_fine = numpy.asarray(y_distance_fine.take(down_samples))

    xticks = numpy.arange(0, 1, 1 / down_sampling_count)
    plt.plot(xticks, y_distance_fine, linestyle='--', marker='*', alpha=0.8)
    plt.plot(xticks, y_distance_coarse, linestyle='--', marker='o', alpha=0.8)
    plt.xticks(xticks)
    plt.title('Result of the Coarse-to-Fine Model')
    plt.xlabel('CDF')
    plt.ylabel('Error (meters)')
    plt.legend(['Fine', 'Coarse'])
    plt.savefig('regression_coarse-to-fine_bonus.pdf')

    print(f'Coarse Classification Model: {calculate_area(y_distance_coarse):.2f}m')
    print(f'Fine Regression Model: {calculate_area(y_distance_fine):.2f}m')

    master_station_group = data_df.groupby(['RNCID_1', 'CellID_1'])
    group_count = master_station_group['CellID_1'].count() \
        .rename(columns={'RNCID_1': 'RNCID', 'CellID_1': 'CellID'}) \
        .reset_index() \
        .rename(columns={'RNCID_1': 'RNCID', 'CellID_1': 'CellID', 0: 'Count'})

    pandas.options.mode.chained_assignment = None  # supress warning on chained assignment


    clf = RandomForestRegressor(max_depth=10, n_estimators=100)
    round_count = 10
    down_sampling_count = 10
    model_error = {}
    plt.style.use('ggplot')

    for _, row in group_count.iterrows():
        master_station_data = data_df[numpy.logical_and(
            data_df['RNCID_1'] == row['RNCID'], data_df['CellID_1'] == row['CellID'])].copy()
        X_station, y_station = prepare_dataset_with_station(master_station_data)

        current_station = station_df[numpy.logical_and(
            station_df['RNCID'] == row['RNCID'], station_df['CellID'] == row['CellID'])]

        station_latitude = float(current_station['Latitude'])
        station_longitude = float(current_station['Longitude'])

        # X_station = X_station.drop(columns=['Latitude_1', 'Longitude_1'])
        y_station['Latitude'] -= station_latitude
        y_station['Longitude'] -= station_longitude

        for i in range(2, 8):
            X_station[f'Latitude_{i}'] = X_station[f'Latitude_{i}'].replace(-1, float('nan'))
            X_station[f'Latitude_{i}'] -= station_latitude

            X_station[f'Longitude_{i}'] = X_station[f'Longitude_{i}'].replace(-1, float('nan'))
            X_station[f'Longitude_{i}'] -= station_longitude
        X_station = X_station.fillna('-10000')

        round_error = numpy.repeat(0.0, down_sampling_count)
        round_time_elapsed = 0
        for i in range(round_count):
            X_train, X_test, y_train, y_test = train_test_split(X_station, y_station, test_size=0.2)

            now = time.time()
            clf.fit(X_train, y_train['Longitude'])
            y_predict_longitude = clf.predict(X_test)
            y_predict_longitude += station_longitude

            clf.fit(X_train, y_train['Latitude'])
            y_predict_latitude = clf.predict(X_test)
            y_predict_latitude += station_latitude
            round_time_elapsed += time.time() - now

            y_test['Latitude'] += station_latitude
            y_test['Longitude'] += station_longitude
            y_distance = haversine_vec(y_predict_latitude, y_predict_longitude,
                                       y_test['Latitude'], y_test['Longitude'])
            y_distance *= 1000  # unit: meters
            y_distance = y_distance.sort_values()
            down_samples = numpy.linspace(0, y_distance.size - 1, down_sampling_count, dtype=int)
            y_distance = numpy.asarray(y_distance.take(down_samples))

            round_error += y_distance

        model_error[(row['RNCID'], row['CellID'])] = round_error / round_count
        round_time_elapsed /= round_count
        print(f'RNCID: {row["RNCID"]} CellID: {row["CellID"]} '
              f'Avg Error: {calculate_area(round_error / round_count):.2f}m '
              f'({round_time_elapsed:.2f}s)')

    y_distances = numpy.asarray(reduce(lambda x, y: x + y, model_error.values()))
    y_distances /= group_count.shape[0]

    xticks = numpy.arange(0, 1, 1 / down_sampling_count)
    plt.plot(xticks, y_distances, linestyle='--', marker='^', alpha=0.8)
    plt.xticks(xticks)
    plt.title('Result of the Regression Model')
    plt.xlabel('CDF')
    plt.ylabel('Error (meters)')
    plt.legend(['Random Forest Regressor'])
    plt.savefig('regression.png')

    print(f'(Overall) Random Forest Regression: {calculate_area(y_distances):.2f}m')

    model_mid_error = {k: (v[4] + v[5]) / 2 for k, v in model_error.items()}
    model_mid_error = sorted(model_mid_error.items(), key=itemgetter(1), reverse=True)

    k = math.ceil(group_count.shape[0] * 0.2)

    top_k_minus = model_mid_error[:k]
    top_k_plus = model_mid_error[-k:]
    top_k_mixed = list(zip(reversed(top_k_plus), top_k_minus))
    raw_groups = {(row['RNCID'], row['CellID']) for _, row in group_count.iterrows()} \
                 - {i[0] for i in top_k_minus} - {i[0] for i in top_k_plus}

    clf = RandomForestRegressor(max_depth=10, n_estimators=100)

    round_count = 10
    down_sampling_count = 10
    mixed_model_error = {}
    plt.style.use('ggplot')

    # groups without mixture
    for rncid, cellid in raw_groups:
        master_station_data = data_df[numpy.logical_and(
            data_df['RNCID_1'] == rncid, data_df['CellID_1'] == cellid)].copy()
        X_station, y_station = prepare_dataset_with_station(master_station_data)

        current_station = station_df[numpy.logical_and(
            station_df['RNCID'] == rncid, station_df['CellID'] == cellid)]
        station_latitude = float(current_station['Latitude'])
        station_longitude = float(current_station['Longitude'])

        y_station['Latitude'] -= station_latitude
        y_station['Longitude'] -= station_longitude

        for i in range(2, 8):
            X_station[f'Latitude_{i}'] = X_station[f'Latitude_{i}'].replace(-1, float('nan'))
            X_station[f'Latitude_{i}'] -= station_latitude

            X_station[f'Longitude_{i}'] = X_station[f'Longitude_{i}'].replace(-1, float('nan'))
            X_station[f'Longitude_{i}'] -= station_longitude
        X_station = X_station.fillna('-10000')

        round_error = numpy.repeat(0.0, down_sampling_count)
        round_time_elapsed = 0
        for i in range(round_count):
            X_train, X_test, y_train, y_test = train_test_split(X_station, y_station, test_size=0.2)

            now = time.time()
            clf.fit(X_train, y_train['Longitude'])
            y_predict_longitude = clf.predict(X_test)
            y_predict_longitude += station_longitude

            clf.fit(X_train, y_train['Latitude'])
            y_predict_latitude = clf.predict(X_test)
            y_predict_latitude += station_latitude
            round_time_elapsed += time.time() - now

            y_test['Latitude'] += station_latitude
            y_test['Longitude'] += station_longitude
            y_distance = haversine_vec(y_predict_latitude, y_predict_longitude,
                                       y_test['Latitude'], y_test['Longitude'])
            y_distance *= 1000  # unit: meters
            y_distance = y_distance.sort_values()
            down_samples = numpy.linspace(0, y_distance.size - 1, down_sampling_count, dtype=int)
            y_distance = numpy.asarray(y_distance.take(down_samples))

            round_error += y_distance

        mixed_model_error[(rncid, cellid)] = round_error / round_count
        round_time_elapsed /= round_count
        print(f'RNCID: {rncid} CellID: {cellid} '
              f'Avg Error: {calculate_area(round_error / round_count):.2f}m '
              f'({round_time_elapsed:.2f}s)')

    # mixed groups of top-k-plus and top-k-minus
    for ((plus_rncid, plus_cellid), _), ((minus_rncid, minus_cellid), _) in top_k_mixed:
        master_station_data = data_df[numpy.logical_or(
            numpy.logical_and(
                data_df['RNCID_1'] == plus_rncid, data_df['CellID_1'] == plus_cellid),
            numpy.logical_and(
                data_df['RNCID_1'] == minus_rncid, data_df['CellID_1'] == minus_cellid))
        ].copy()

        X_station, y_station = prepare_dataset_with_station(master_station_data)
        y_station_longitude = numpy.asarray(y_station['Longitude']) - X_station['Longitude_1']
        y_station_latitude = numpy.asarray(y_station['Latitude']) - X_station['Latitude_1']
        y_station = pandas.DataFrame(numpy.column_stack((y_station_latitude, y_station_longitude)),
                                 columns=['Latitude', 'Longitude'])

        for i in range(2, 8):
            X_station[f'Latitude_{i}'] = X_station[f'Latitude_{i}'].replace(-1, float('nan'))
            X_station[f'Latitude_{i}'] -= X_station['Latitude_1']

            X_station[f'Longitude_{i}'] = X_station[f'Longitude_{i}'].replace(-1, float('nan'))
            X_station[f'Longitude_{i}'] -= X_station['Longitude_1']

        X_station = X_station.fillna('-10000')

        round_error = numpy.repeat(0.0, down_sampling_count)
        round_time_elapsed = 0
        for _ in range(round_count):
            X_train, X_test, y_train, y_test = train_test_split(X_station, y_station, test_size=0.2)

            now = time.time()
            clf.fit(X_train, y_train['Longitude'])
            y_predict_longitude = clf.predict(X_test)
            y_predict_longitude += station_longitude

            clf.fit(X_train, y_train['Latitude'])
            y_predict_latitude = clf.predict(X_test)
            y_predict_latitude += station_latitude
            round_time_elapsed += time.time() - now

            y_test['Latitude'] += station_latitude
            y_test['Longitude'] += station_longitude
            y_distance = haversine_vec(y_predict_latitude, y_predict_longitude,
                                       y_test['Latitude'], y_test['Longitude'])
            y_distance *= 1000  # unit: meters
            y_distance = y_distance.sort_values()
            down_samples = numpy.linspace(0, y_distance.size - 1, down_sampling_count, dtype=int)
            y_distance = numpy.asarray(y_distance.take(down_samples))

            round_error += y_distance

        mixed_model_error[((plus_rncid, plus_cellid), (minus_rncid, minus_cellid))] = round_error / round_count
        round_time_elapsed /= round_count
        print(f'Plus RNCID: {plus_rncid} CellID: {plus_cellid} '
              f'Minus RNCID: {minus_rncid} CellID: {minus_cellid} '
              f'Avg Error: {calculate_area(round_error / round_count):.2f}m '
              f'({round_time_elapsed:.2f}s)')

    y_distances = numpy.asarray(reduce(lambda x, y: x + y, mixed_model_error.values()))
    y_distances /= group_count.shape[0]

    xticks = numpy.arange(0, 1, 1 / down_sampling_count)
    plt.plot(xticks, y_distances, linestyle='--', marker='^', alpha=0.8)
    plt.xticks(xticks)
    plt.title('Result of the Regression Model After Group Mixture')
    plt.xlabel('CDF')
    plt.ylabel('Error (meters)')
    plt.legend(['Random Forest Regressor'])
    plt.savefig('regression_mixture.pdf')

    print(f'(Overall) Random Forest Regression: {calculate_area(y_distances):.2f}m')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    preprocessing()
    rasterize()


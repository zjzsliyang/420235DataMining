import os
import math
import pandas
import pickle
import logging
from datetime import datetime
from collections import defaultdict

preprocessed_path = os.getcwd() + '/preprocessed_trade.pkl'


def preprocessing():
    start = datetime.now()

    types = {'uid': str, 'vipno': str, 'pluno': str, 'spec': str, 'pkunit': str,
             'dptno': str, 'bndname': str}
    df = pandas.read_csv('../../data/reco_data/trade_new.csv',
                         dtype=types)
    df['sldatime'] = pandas.to_datetime(df['sldatime'])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[df.isdel == 0]
    df['bndno'].fillna(-1, inplace=True)
    df['gender'] = [1 if '男' in x else 0 for x in df['cmrid']]

    def condition(x):
        if '[18 以下]' in x:
            return 1
        elif '[18 - 25]' in x:
            return 2
        elif '[26 - 30]' in x:
            return 3
        elif '[31 - 45]' in x:
            return 4
        elif '[45 以上]' in x:
            return 5

    df['age_range'] = [condition(x) for x in df['cmrid']]
    df.drop(['pno', 'cno', 'cmrid', 'bcd', 'id', 'disamt', 'mdocno', 'isdel'],
            axis=1, inplace=True)
    alias = {'uid': 'order_id', 'sldatime': 'order_time', 'vipno': 'vip_no',
             'pluno': 'item_id', 'pluname': 'item_name',
             'spec': 'item_specification', 'pkunit': 'item_unit',
             'dptno': 'category_no', 'bndno': 'brand_no',
             'bndname': 'brand_name', 'qty': 'quantity', 'amt': 'amount',
             'ismmx': 'promotion', 'mtype': 'promotion_type'}
    df.rename(columns=alias, inplace=True)

    f = open(preprocessed_path, 'wb')
    pickle.dump(df, f)
    logging.info('time cost in preprocessing: ' + str(datetime.now() - start))


def features():
    f = open(preprocessed_path, 'rb')
    df = pickle.load(f)
    print(df)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    preprocessing()
    features()

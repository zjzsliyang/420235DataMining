import logging
import matplotlib
from lshash.lshash import LSHash


def knn(df, k, coefficient):
    hash_size = int(coefficient * df.shape[1])
    lsh = LSHash(hash_size, input_dim=df.shape[0])
    for vipno in df:
        lsh.index(df[vipno], extra_data=vipno)
    random_column = df[df.columns.to_series().sample(1)]
    random_vip = random_column.columns.values[0]
    logging.info('random vipno: {}'.format(random_vip))
    res = lsh.query(random_column.values.flatten())[0: k + 1]
    print('vipno in ranked order using kNN(k = {}):'.format(k))
    knns = []
    for item in res:
        print(item[0][1])
        knns.append(item[0][1])
    return random_vip, knns

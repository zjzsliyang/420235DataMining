import numpy
import pandas
from collections import defaultdict


def brief(object1: str, object2='_'):
    return '_' + object1.split('_')[0] + '_' + object2.split('_')[0] + '_'


# PART I: count/ratio
def count(df: pandas.DataFrame, features: defaultdict):
    period = ['monthly', 'whole']
    objects = ['vip_no', 'brand_no', 'category_no', 'item_id']
    duo_objs = [(0, 1), (0, 2), (0, 3), (1, 2)]
    norms = {'count': list.__len__, 'amount': sum, 'purchase_day': (lambda x: len(set(x)))}
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for index, row in df.iterrows():
        for norm in norms.keys():
            for obj in objects:
                raw[period[0] + brief(obj) + norm][row[obj]][row['order_time'].month].append(row[norm])
            for duo_index in duo_objs:
                raw[period[0] + brief(objects[duo_index[0]], objects[duo_index[1]]) + norm][
                    objects[duo_index[0]] + '_' + objects[duo_index[1]]][row['order_time'].month].append(row[norm])

    for norm in norms.keys():
        for obj in objects:
            for obj_key, obj_value in raw[period[0] + brief(obj) + norm].items():
                tmp = []
                for time, item in obj_value.items():
                    tmp += item
                    features[period[0] + brief(obj) + norm][obj_key][time] = norms[norm](item)
                features[period[1] + brief(obj) + norm][obj_key][period[1]] = norms[norm](tmp)
        for duo_index in duo_objs:
            for obj_key, obj_value in raw[
                period[0] + brief(objects[duo_index[0]], objects[duo_index[1]]) + norm].items():
                tmp = []
                for time, item in obj_value.items():
                    tmp += item
                    features[period[0] + brief(objects[duo_index[0]], objects[duo_index[1]]) + norm][obj_key][time] = \
                        norms[norm](item)
                features[period[1] + brief(objects[duo_index[0]], objects[duo_index[1]]) + norm][obj_key][period[1]] = \
                    norms[norm](tmp)


def product_diversity(df: pandas.DataFrame, features: defaultdict):
    period = ['monthly', 'whole']
    objects = ['vip_no', 'brand_no', 'category_no', 'item_id']
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for index, row in df.iterrows():
        for obj in objects[1:]:
            raw[period[0] + brief(obj) + 'unique'][row[objects[0]]][row['order_time'].month].add(row[obj])
        raw[period[0] + brief(objects[1], object[2]) + 'unique'][(row[objects[1]], row[objects[2]])][
            row['order_time'].month].add(row[objects[3]])

    for obj in objects[1:]:
        for obj_key, obj_value in raw[period[0] + brief(obj) + 'unique'].items():
            tmp = set()
            for time, item in obj_value.items():
                tmp.update(item)
                features[period[0] + brief(obj) + 'unique'][obj_key][time] = len(item)
            features[period[1] + brief(obj) + 'unique'][obj_key][period[1]] = len(tmp)
        for obj_key, obj_value in raw[period[0] + brief(objects[1], object[2]) + 'unique'].items():
            tmp = set()
            for time, item in obj_value.items():
                tmp.update(item)
                features[period[0] + brief(objects[1], object[2]) + 'unique'][obj_key][time] = len(item)
            features[period[1] + brief(objects[1], object[2]) + 'unique'][obj_key][period[1]] = len(tmp)


def penetration(df: pandas.DataFrame, features: defaultdict):
    period = ['monthly', 'whole']
    objects = ['vip_no', 'brand_no', 'category_no', 'item_id']
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for index, row in df.iterrows():
        for obj in objects[1:]:
            raw[period[0] + brief(obj) + 'unique_vip'][row[obj]][row['order_time'].month].add(row[objects[0]])

    for obj in objects[1:]:
        for obj_key, obj_value in raw[period[0] + brief(obj) + 'unique_vip'].items():
            tmp = set()
            for time, item in obj_value.items():
                tmp.update(item)
                features[period[0] + brief(obj) + 'unique_vip'][obj_key][time] = len(item)
            features[period[1] + brief(obj) + 'unique_vip'][obj_key][period[1]] = len(tmp)


# PART II: AGG feature
def month_agg(df: pandas.DataFrame, features: defaultdict):
    agg = {'mean': numpy.mean, 'std': numpy.std, 'max': numpy.max, 'median': numpy.median}

    for raw_feature in features.keys():
        if 'month' in raw_feature:
            for obj_key, obj_value in features[raw_feature].item():
                agg_feature = raw_feature.replace('month', 'month_agg')
                for agg_name, agg_method in agg.items():
                    features[agg_feature][obj_key][agg_name] = agg_method(numpy.array(obj_value.values()))


def user_agg(df: pandas.DataFrame, features: defaultdict):
    objects = ['vip_no', 'brand_no', 'category_no', 'item_id']
    norms = {'count': list.__len__, 'amount': sum, 'purchase_day': (lambda x: len(set(x)))}
    agg = {'mean': numpy.mean, 'std': numpy.std, 'max': numpy.max, 'median': numpy.median}
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tmp = defaultdict(lambda: defaultdict(dict))

    for index, row in df.iterrows():
        for norm in norms.keys():
            for obj in objects[1:]:
                raw['user_agg' + brief(obj) + norm][row[obj]][row[objects[0]]].append(row[norm])

    for norm in norms.keys():
        for obj in objects[1:]:
            agg_feature = 'user_agg' + brief(obj) + norm
            for obj_key, obj_value in raw[agg_feature].items():
                for user, item in obj_value.items():
                    tmp[agg_feature][obj_key][user] = norms[norm](item)
            for obj_key, obj_value in tmp[agg_feature].items():
                for agg_name, agg_method in agg.items():
                    features[agg_feature][obj_key][agg_name] = agg_method(numpy.array(obj_value.values()))


def obj_agg(df: pandas.DataFrame, features: defaultdict):
    # TODO:
    agg = {'mean': numpy.mean, 'std': numpy.std, 'max': numpy.max, 'median': numpy.median}


# PART III: last week/ last month feature
def recent_feature(df: pandas.DataFrame, features: defaultdict):
    period = ['last_month', 'last_week']
    month_begin = df['order_time'].max - pandas.offsets.relativedelta(months=1)
    week_begin = df['order_time'].max - pandas.offsets.relativedelta(weeks=1)
    last_month_df = df.loc[(df['order_time'] > month_begin)]
    last_week_df = df.loc[(df['order_time'] > week_begin)]
    # TODO: partial function


# PART IV: complex feature
def trend(df: pandas.DataFrame, features: defaultdict):
    # regression using least squares, using slope as trend

    # calculate normalized mean of the last month and the previous months
    pass


def repeat_feature(df: pandas.DataFrame, features: defaultdict):
    pass


def market_share(df: pandas.DataFrame, features: defaultdict):
    pass


def similarity(df: pandas.DataFrame, features: defaultdict):
    pass


def generate_feature(df: pandas.DataFrame, features: defaultdict):
    # PART I: count/ratio
    count(df, features)
    product_diversity(df, features)
    penetration(df, features)

    # PART II: AGG feature
    month_agg(df, features)
    user_agg(df, features)
    obj_agg(df, features)

    # PART III: last week/ last month feature
    recent_feature(df, features)

    # PART IV: complex feature
    trend(df, features)
    repeat_feature(df, features)
    market_share(df, features)

import pandas
from collections import defaultdict


def brief(object1: str, object2='_'):
    return '_' + object1.split('_')[0] + '_' + object2.split('_')[0] + '_'

# PART I: count/ratio
def count(df: pandas.DataFrame, features: defaultdict):
    period = ['monthly', 'whole']
    objects = ['vip_no', 'brand_no', 'category_no', 'item_id']
    duo_objs = [(0, 1), (0, 2), (0, 3), (1, 2)]
    norms = ['count', 'amount', 'purchase_day']
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for index, row in df.iterrows():
        for norm in norms[1:]:
            for obj in objects:
                raw[period[0] + brief(obj) + norm][row[obj]][
                    row['order_time'].month].append(row[norm])
            for duo_index in duo_objs:
                raw[period[0] + brief(objects[duo_index[0]],
                                           objects[duo_index[0]]) + norm][
                    objects[duo_index[0]] + '_' + objects[duo_index[1]]][
                    row['order_time'].month].append(row[norm])

    # TODO: calculate the whole


def product_diversity(df: pandas.DataFrame, features: defaultdict):
    period = ['monthly', 'whole']
    objects = ['vip_no', 'brand_no', 'category_no', 'item_id']
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for index, row in df.iterrows():
        for obj in objects[1:]:
            raw[period[0] + brief(obj) + 'unique'][row[objects[0]]][row['order_time'].month].add(row[obj])

        raw[period[0] + brief(objects[1], object[2]) + 'unique'][(row[objects[1]], row[objects[2]])][row['order_time'].month].add(row[objects[3]])


def penetration(df: pandas.DataFrame, features: defaultdict):
    period = ['monthly', 'whole']
    objects = ['vip_no', 'brand_no', 'category_no', 'item_id']
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for index, row in df.iterrows():
        for obj in objects[1:]:
            raw[period[0] + brief(obj) + 'unique'][row[obj]][row['order_time'].month].add(row[objects[0]])


# PART II: AGG feature
def month_agg(df: pandas.DataFrame, features: defaultdict):
    pass


def user_agg(df: pandas.DataFrame, features: defaultdict):
    pass


def obj_agg(df: pandas.DataFrame, features: defaultdict):
    pass


# PART III: last week/ last month feature
def recent_feature(df: pandas.DataFrame, features: defaultdict):
    pass


# PART IV: complex feature
def trend(df: pandas.DataFrame, features: defaultdict):
    pass


def repeat_feature(df: pandas.DataFrame, features: defaultdict):
    pass


def market_share(df: pandas.DataFrame, features: defaultdict):
    pass

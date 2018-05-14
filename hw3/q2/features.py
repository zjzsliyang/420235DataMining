import pandas
from collections import defaultdict


# PART I: count/ratio
def count(df: pandas.DataFrame, features: defaultdict):
    period = ['monthly', 'whole']
    objects = ['vip_no', 'brand_no', 'category_no', 'item_id']
    duo_objs = [(0, 1), (0, 2), (0, 3), (1, 2)]
    norms = ['count', 'amount', 'purchase_day']

    def brief(object1: str, object2='_'):
        return '_' + object1.split('_')[0] + '_' + object2.split('_')[0] + '_'

    for index, row in df.iterrows():
        for norm in norms:
            for obj in objects:
                features[period[0] + brief(obj) + norm][row[obj]][
                    row['order_time'].month].append(row[norm])
            for duo_index in duo_objs:
                features[period[0] + brief(objects[duo_index[0]],
                                           objects[duo_index[0]]) + norm][
                    objects[duo_index[0]] + '_' + objects[duo_index[1]]][
                    row['order_time'].month].append(
                    1 if norm is 'count' else row[norm])

    # TODO: calculate the whole


def product_diversity(df: pandas.DataFrame, features: defaultdict):
    pass


def penetration(df: pandas.DataFrame, features: defaultdict):
    pass


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

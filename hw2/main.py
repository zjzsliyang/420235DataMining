import sys
import pandas
import logging
import pyfpgrowth
from hw2.src.PrefixSpan import prefixSpan, SquencePattern, print_patterns


def find_frequent_patterns(sequential: bool, combined: bool, threshold: int,
                           item_no: str,
                           is_new: bool):
    df = pandas.read_csv(
        '../data/reco_data/trade_new.csv') if is_new else pandas.read_csv(
        '../data/reco_data/trade.csv')
    sldat = 'sldatime' if is_new else 'sldat'

    if not sequential:
        if combined:
            data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                lambda x: x.sort_values(by=sldat, ascending=True).head(
                    int(x[item_no].count() * 0.6)))[item_no].groupby(
                'vipno').apply(
                set).as_matrix()
        else:
            data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                lambda x: x.sort_values(by=sldat, ascending=True).head(
                    int(x[item_no].count() * 0.6))).groupby(sldat)[
                item_no].apply(
                set).as_matrix()
        patterns = pyfpgrowth.find_frequent_patterns(data, threshold)
        print(patterns)
        print(len(patterns))
    else:
        if combined:
            data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                lambda x: x.sort_values(by=sldat, ascending=True).head(
                    int(x[item_no].count() * 0.6))).groupby(['vipno', sldat])[
                item_no].apply(set).apply(list).groupby('vipno').apply(
                list).tolist()
        else:
            data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                lambda x: x.sort_values(by=sldat, ascending=True).head(
                    int(x[item_no].count() * 0.6))).groupby(sldat)[
                item_no].apply(
                set).apply(list).as_matrix()
            data = [[[j] for j in i] for i in data]
        patterns = prefixSpan(SquencePattern([], sys.maxsize), data, threshold)
        print_patterns(patterns)
        print(len(patterns))


def main():
    logging.basicConfig(level=logging.INFO)
    find_frequent_patterns(sequential=True, combined=False, threshold=2,
                           item_no='dptno', is_new=True)


if __name__ == '__main__':
    main()

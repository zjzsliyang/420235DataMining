import sys
import pandas
import logging
import operator
import pyfpgrowth
from statistics import mean
from hw2.src.PrefixSpan import prefixSpan, SquencePattern, print_patterns


def find_frequent_patterns(sequential: bool, combined: bool, threshold: int,
                           item_no: str, is_new: bool):
    df = pandas.read_csv(
        '../data/reco_data/trade_new.csv') if is_new else pandas.read_csv(
        '../data/reco_data/trade.csv')
    sldat = 'sldatime' if is_new else 'sldat'

    if not sequential:
        if combined:
            data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                lambda x: x.sort_values(by=sldat, ascending=True).head(
                    int(x[item_no].count() * 0.6)))[item_no].dropna().groupby(
                'vipno').apply(set).as_matrix()
        else:
            data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                lambda x: x.sort_values(by=sldat, ascending=True).head(
                    int(x[item_no].count() * 0.6))).dropna().groupby(sldat)[
                item_no].apply(set).as_matrix()
        patterns = pyfpgrowth.find_frequent_patterns(data, threshold)
        logging.info('len of frequent patterns: {}'.format(len(patterns)))
        print(patterns)
        return patterns
    else:
        if combined:
            data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                lambda x: x.sort_values(by=sldat, ascending=True).head(
                    int(x[item_no].count() * 0.6))).dropna().groupby(
                ['vipno', sldat])[item_no].apply(set).apply(list).groupby(
                'vipno').apply(list).tolist()
        else:
            data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                lambda x: x.sort_values(by=sldat, ascending=True).head(
                    int(x[item_no].count() * 0.6))).dropna().groupby(sldat)[
                item_no].apply(set).apply(list).as_matrix()
            data = [[[j] for j in i] for i in data]
        patterns = prefixSpan(SquencePattern([], sys.maxsize), data, threshold)
        logging.info('len of frequent patterns: {}'.format(len(patterns)))
        print_patterns(patterns)
        return patterns


def verification(patterns, sequential: bool, item_no: str, is_new: bool):
    df = pandas.read_csv(
        '../data/reco_data/trade_new.csv') if is_new else pandas.read_csv(
        '../data/reco_data/trade.csv')
    sldat = 'sldatime' if is_new else 'sldat'

    olders = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
        lambda x: x.sort_values(by=sldat, ascending=False).head(
            int(x[item_no].count() * 0.6))).groupby('vipno')[item_no].apply(
        lambda x: x.tolist()).to_dict()
    newers = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
        lambda x: x.sort_values(by=sldat, ascending=False).head(
            int(x[item_no].count() * 0.4))).groupby(['vipno', sldat])[item_no].apply(
        lambda x: x.tolist()).to_dict()

    if not sequential:
        sort_pattern = sorted(patterns.items(), key=operator.itemgetter(1),
                              reverse=True)
        reco = {}
        precision = {}
        recall = {}
        for pattern in sort_pattern:
            for vipno, old_tran in olders.items():
                remaining = {*pattern[0]} - {*pattern[0]}.intersection(
                    old_tran)
                if len(remaining) == 1 and len(pattern) > 1:
                    reco[vipno] = reco.get(vipno, set())
                    reco[vipno].add(lambda x: x in remaining)
        for index, new_tran in newers.items():
            vipno = index[0]
            hit = len(set(reco[vipno]).intersection(new_tran))
            precision[vipno] = hit / len(new_tran)
            recall[vipno] = hit / len(reco[vipno])

        print('precision: {}'.format(mean(precision.values())))
        print('recall: {}'.format(mean(recall.values())))


def main():
    logging.basicConfig(level=logging.INFO)
    patterns = find_frequent_patterns(sequential=False, combined=True,
                                      threshold=2, item_no='pluno', is_new=True)
    verification(patterns=patterns, sequential=False, item_no='pluno', is_new=True)


if __name__ == '__main__':
    main()

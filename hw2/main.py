import pandas
import logging
import pyfpgrowth


def find_frequent_patterns(combined: bool, threshold: int, item_no: str,
                           is_new: bool):
    df = pandas.read_csv(
        '../data/reco_data/trade_new.csv') if is_new else pandas.read_csv(
        '../data/reco_data/trade.csv')
    sldat = 'sldatime' if is_new else 'sldat'
    if combined:
        data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
            lambda x: x.sort_values(by=sldat, ascending=True).head(
                int(x[item_no].count() * 0.6)))[item_no].groupby('vipno').apply(
            set).as_matrix()
    else:
        data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
            lambda x: x.sort_values(by=sldat, ascending=True).head(
                int(x[item_no].count() * 0.6))).groupby(sldat)[item_no].apply(
            set).as_matrix()
        print(data)
    patterns = pyfpgrowth.find_frequent_patterns(data, threshold)
    print(patterns)
    print(len(patterns))


def main():
    logging.basicConfig(level=logging.INFO)
    find_frequent_patterns(combined=False, threshold=2, item_no='pluno',
                           is_new=True)


if __name__ == '__main__':
    main()

import csv
import pandas
import logging
import matplotlib.pyplot
from hw1.q1.knn import knn
import palettable.colorbrewer.qualitative
from mpl_toolkits.mplot3d import Axes3D


def read_data():
    trade = {}
    with open('../data/reco_data/trade.csv', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        for row in reader:
            vipno = row[header.index('vipno')]
            pluno = row[header.index('pluno')]
            amt = float(row[header.index('amt')])
            if vipno not in trade:
                trade[vipno] = {}
            trade[vipno][pluno] = amt + trade[vipno].get(pluno, 0)
    # DataFrame structure:
    #          vipno
    # pluno    amt
    df = pandas.DataFrame(trade).fillna(0).round(0)
    df.columns.name = 'vipno'
    df.index.name = 'pluno'
    logging.info('DataFrame shape: {}'.format(df.shape))
    logging.debug('DataFrame info: {}'.format(df.info(verbose=False)))

    plot_raw_data(df)
    return df


def plot_raw_data(df):
    df.to_csv('res/trade.csv')
    x = []
    y = []
    z = []
    fig = matplotlib.pyplot.figure().gca(projection='3d')
    i = 0
    les10 = 0
    les100 = 0
    les1000 = 0
    more1000 = 0
    for index, row in df.iterrows():
        for col in df.columns:
            if row[col] != 0:
                # x.append(index)
                # y.append(col)
                x.append(i)
                y.append(i)
                z.append(row[col])
                if row[col] < 10:
                    les10 += 1
                elif row[col] < 100:
                    les100 += 1
                elif row[col] < 1000:
                    les1000 += 1
                else:
                    more1000 += 1
                i += 1
    fig.scatter(x, y, z)
    logging.info('point No. which are not zero: {}'.format(len(z)))
    matplotlib.pyplot.savefig('res/raw.jpg', dpi=100)

    matplotlib.pyplot.figure(figsize=(8,6))
    matplotlib.pyplot.pie([les10, les100, les1000, more1000], colors=palettable.colorbrewer.qualitative.Pastel1_4.mpl_colors, labels=['[0, 10)', '[10, 100)', '[100, 1000)', '[1000, inf)'])
    matplotlib.pyplot.axis('equal')
    matplotlib.pyplot.savefig('res/raw2.jpg', dpi=100)
    matplotlib.pyplot.show()


def main():
    logging.basicConfig(level=logging.DEBUG)

    df = read_data()

    # Problem I
    coefficients = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    ks = [1, 2, 3, 4, 5]
    random_vip, knns = knn(df, ks[4], coefficients[0])

    # Problem II
    # k = kmeans(df, random_vip, knns)

    # Problem III
    # eps = dbscan(df, random_vip, knns)

    # Problem IV
    # gmm(df, k, eps, random_vip, knns)


if __name__ == '__main__':
    main()

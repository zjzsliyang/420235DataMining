## Frequent Pattern with Sequence

*1452669, Yang LI, May 5*

### Data Processing

- Requirement: do not forget to add python path in the shell.

  ```shell
  export PYTHONPATH=$PYTHONPATH:PATHTOTHEFOLDER
  ```

- The frequent pattern result is `bi.txt` in the `b` subfolder.  (with `threshold=2, item_no='pluno', is_new=True`)

- Read data as DataFrame, and using functional programming to process data.

  ```python
  df = pandas.read_csv(
          '../data/reco_data/trade_new.csv') if is_new else pandas.read_csv(
          '../data/reco_data/trade.csv')
  data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                  lambda x: x.sort_values(by=sldat, ascending=True).head(
                      int(x[item_no].count() * 0.6))).dropna().groupby(sldat)[
                  item_no].apply(set).apply(list).as_matrix()
  data = [[[j] for j in i] for i in data]
  ```

### Analysis

lenth of pattern in `trade.csv`:

| Support | 64   | 32   | 16   | 8    | 4    | 2    |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- |
| dptno   | 14   | 35   | 96   | 251  | 761  | 3529 |
| pluno   | 7    | 21   | 50   | 127  | 393  | 1893 |
| bndno   | 3    | 9    | 11   | 33   | 113  | 413  |

lenth of pattern in `trade_new.csv`:

| Support | 64   | 32   | 16   | 8    | 4    | 2     |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| dptno   | 34   | 99   | 247  | 648  | 2021 | 10234 |
| pluno   | 10   | 38   | 108  | 298  | 898  | 4069  |
| bndno   | 5    | 13   | 55   | 132  | 384  | 1661  |

time cost in different support:

- new

  ![](../res/bi_new.jpg)

- old

  ![](../res/bi_old.jpg)

### Performance

##### Time & Space Complexity in Theory

We use PrefixSpan algorithm to implement the Frequent Pattern Mining with Sequence, it can divide into several parts.

- Find length-1 sequential patterns

  The given sequence S is scanned to get item (prefix) that occurred frequently in S. For the number of time that item occurs is equal to length-l of that item. Length-l is given by notation \<pattern>: \<count>.

- Divide search soace

  Based on the prefix that derived from first step, the whole sequential pattern set is partitioned in this phase. 

- FInd subsets of sequential patterns

  The projected databases are constructed and sequential patterns are mined from these databases. Only local frequent sequences are explored in projected databases so as to expand the sequential patterns. The cost for constructing projected database is quite high. Bi-level projection and pseudo-projection methods are used to reduce this cost which ultimately increases the algorithm’s efficiency.

##### Benchmark in Practice

![](../res/biline.png)

![](../res/bimem.png)

### Screenshot

![](../res/bi.png)

![](../res/biprofile.png)
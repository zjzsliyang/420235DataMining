## Frequent Pattern with Sequence

*1452669, Yang LI, May 5*

### Data Processing

- Requirement: do not forget to add python path in the shell.

  ```shell
  export PYTHONPATH=$PYTHONPATH:PATHTOTHEFOLDER
  ```

- The frequent pattern result is `bi.txt` in the `b` subfolder.

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

### Performance

##### Time & Space Complexity in Theory



##### Benchmark in Practice

![](../res/biline.png)

![](../res/bimem.png)

### Screenshot

![](../res/bi.png)

![](../res/biprofile.png)
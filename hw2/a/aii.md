## Frequent Pattern without Sequence

*1452669, Yang LI, May 5*

### Data Processing

- Requirement: do not forget to add python path in the shell.

  ```shell
  export PYTHONPATH=$PYTHONPATH:PATHTOTHEFOLDER
  ```

- The frequent pattern result is `aii.txt` in the `a` subfolder.

- Read data as DataFrame, and using functional programming to process data.

  ```python
  df = pandas.read_csv(
          '../data/reco_data/trade_new.csv') if is_new else pandas.read_csv(
          '../data/reco_data/trade.csv')
  data = df[[item_no, 'vipno', sldat]].groupby('vipno').apply(
                  lambda x: x.sort_values(by=sldat, ascending=True).head(
                      int(x[item_no].count() * 0.6)))[item_no].dropna().groupby(
                  'vipno').apply(set).as_matrix()
  ```

### Performance

##### Time & Space Complexity in Theory



##### Benchmark in Practice

![](../res/aiiline.png)

![](../res/aiimem.png)

### Screenshot

![](../res/aii.png)

![](../res/aiiprofile.png)
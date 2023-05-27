# Order Statistics

The $i$th [Order Statistic](https://en.wikipedia.org/wiki/Order_statistic) of a random sample of size $n$ from a univariate distribution is the $i$th element after sorting in increasing order.
As a special case, the first and $n$th order statistics are the minimum and maximum of the sample, while for odd $n$, the $\lceil \frac{n}{2} \rceil$th entry is the sample median.

Given any univariate distribution and the sample size $n$, we can construct the distribution of its $i$th order statistic:

```@docs
OrderStatistic
```

If we are interested in more than one order statistic, for continuous univariate distributions we can also construct the joint distribution of order statistics:

```@docs
JointOrderStatistics
```

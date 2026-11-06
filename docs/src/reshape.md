# Reshaping distributions

Distributions of array variates such as `MultivariateDistribution`s and
`MatrixDistribution`s can be reshaped.

```@docs
reshape
vec
```

Univariate distributions have scalar variates and cannot be reshaped.
Both `reshape` and `ReshapedDistribution` reject them with an `ArgumentError`.

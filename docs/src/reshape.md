# Reshaping distributions

Distributions of array variates such as `MultivariateDistribution`s and
`MatrixDistribution`s can be reshaped.

```@docs
reshape
vec
```

Univariate distributions have scalar variates and cannot be reshaped.
Both `reshape` and `vec` reject them with an error.

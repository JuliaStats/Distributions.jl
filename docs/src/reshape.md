# Reshaping distributions

Distributions of array variates such as `MultivariateDistribution`s and
`MatrixDistribution`s can be reshaped.

```@docs
reshape
vec
```

A one-element array with the same shape as the reshaped univariate distribution
represents a single observation.
For example, `insupport(reshape(Uniform(), 1), [0.5])` returns `true`.

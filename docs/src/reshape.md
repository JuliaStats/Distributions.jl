# Reshaping distributions

Distributions of array variates such as `MultivariateDistribution`s and
`MatrixDistribution`s can be reshaped.

Reshaping preserves the element type of the wrapped distribution. For example,
reshaping an `MvNormal` with `Float32` elements also produces `Float32` samples.

```@docs
reshape
vec
```

# [Matrix-variate Distributions](@id matrix-variates)

*Matrix-variate distributions* are the distributions whose variate forms are `Matrixvariate` (*i.e* each sample is a matrix). Abstract types for matrix-variate distributions:

## Common Interface

Both distributions implement the same set of methods:

```@docs
size(::MatrixDistribution)
length(::MatrixDistribution)
mean(::MatrixDistribution)
pdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T})
logpdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T})
rand(::MatrixDistribution, ::Int)
```

## Distributions

```@docs
Wishart
InverseWishart
```

## Internal Methods (for creating your own matrix-variate distributions)

```@docs
Distributions._logpdf(d::MatrixDistribution, x::AbstractArray)
```

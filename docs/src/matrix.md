# [Matrix-variate Distributions](@id matrix-variates)

*Matrix-variate distributions* are the distributions whose variate forms are `Matrixvariate` (*i.e* each sample is a matrix). Abstract types for matrix-variate distributions:

```julia
const MatrixDistribution{S<:ValueSupport} = Distribution{Matrixvariate,S}

const DiscreteMatrixDistribution   = Distribution{Matrixvariate, Discrete}
const ContinuousMatrixDistribution = Distribution{Matrixvariate, Continuous}
```

More advanced functionalities related to random matrices can be found in the
[RandomMatrices.jl](https://github.com/JuliaMath/RandomMatrices.jl) package.

## Common Interface

All distributions implement the same set of methods:

```@docs
size(::MatrixDistribution)
length(::MatrixDistribution)
Distributions.rank(::MatrixDistribution)
mean(::MatrixDistribution)
var(::MatrixDistribution)
cov(::MatrixDistribution)
pdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T})
logpdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T})
Distributions._rand!(::AbstractRNG, ::MatrixDistribution, A::AbstractMatrix)
vec(d::MatrixDistribution)
```

## Distributions

```@docs
Wishart
InverseWishart
MatrixNormal
MatrixTDist
MatrixBeta
MatrixFDist
LKJ
```

## Internal Methods (for creating your own matrix-variate distributions)

```@docs
Distributions._logpdf(d::MatrixDistribution, x::AbstractArray)
```

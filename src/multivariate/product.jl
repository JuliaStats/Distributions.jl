import Statistics: mean, var, cov

"""
    Product <: MultivariateDistribution

An N dimensional `MultivariateDistribution` constructed from a vector of N independent
`UnivariateDistribution`s.

```julia
Product(Normal.(randn(10), 1)) # A 10-dimensional Product from 10 independent Normals.
```
"""
struct Product{
    S<:ValueSupport,
    T<:UnivariateDistribution{S},
    V<:AbstractVector{T},
} <: MultivariateDistribution{S}
    v::V
    function Product(v::V) where
        V<:AbstractVector{T} where
        T<:UnivariateDistribution{S} where
        S<:ValueSupport
        return new{S, T, V}(v)
    end
end
length(d::Product) = length(d.v)
_rand!(rng::AbstractRNG, d::Product, x::AbstractVector{<:Real}) =
    broadcast!(dn->rand(rng, dn), x, d.v)
_logpdf(d::Product, x::AbstractVector{<:Real}) =
    sum(n->logpdf(d.v[n], x[n]), 1:length(d))

mean(d::Product) = mean.(d.v)
var(d::Product) = var.(d.v)
cov(d::Product) = Diagonal(var(d))
entropy(d::Product) = sum(entropy, d.v)

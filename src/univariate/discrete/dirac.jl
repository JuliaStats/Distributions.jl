"""
    Dirac(value)

A *Dirac distribution* is parametrized by its only value, and takes its value with probability 1.

```julia
d = Dirac(2.0)  # Dirac distribution with value = 2.
rand(d)         # Always returns the same value
```
"""
struct Dirac{T <: Number} <:
    CountableUnivariateDistribution{CountableSupport{T}}
    value::T
end

rand(::AbstractRNG, d::Dirac) = d.value
pmf(d::Dirac, x) = x ≈ d.value ? 1.0 : 0.0
cmf(d::Dirac, x) = x < d.value ? 0.0 : 1.0
quantile(d::Dirac{T}, p::Real) where T = 0 ≤ p ≤ 1 ? d.value : T(NaN)
minimum(d::Dirac) = d.value
maximum(d::Dirac) = d.value
support(d::Dirac) = [d.value]
insupport(d::Dirac, x) = x == d.value
mean(d::Dirac) = d.value
var(d::Dirac) = 0.0
mode(d::Dirac) = d.value
skewness(d::Dirac) = 0.0
kurtosis(d::Dirac) = 0.0
entropy(d::Dirac) = 0.0
mgf(d::Dirac, t) = exp(t * d.value)
cf(d::Dirac, t) = exp(im * t * d.value)

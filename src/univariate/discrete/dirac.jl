"""
    Dirac(x)

A *Dirac distribution* is parameterized by its only value `x`, and takes its value with probability 1.

```math
P(X = \\hat{x}) = \\begin{cases}
1 & \\quad \\text{for } \\hat{x} = x, \\\\
0 & \\quad \\text{for } \\hat{x} \\neq x.
\\end{cases}
```

```julia
Dirac(2.5)   # Dirac distribution with value x = 2.5
```

External links:

* [Dirac measure on Wikipedia](http://en.wikipedia.org/wiki/Dirac_measure)
"""
struct Dirac{T} <: DiscreteUnivariateDistribution
    value::T
end

Base.eltype(::Type{Dirac{T}}) where {T} = T

insupport(d::Dirac, x::Real) = x == d.value
minimum(d::Dirac) = d.value
maximum(d::Dirac) = d.value
support(d::Dirac) = (d.value,)

#### Properties
mean(d::Dirac) = d.value
var(d::Dirac{T}) where {T} = zero(T)

mode(d::Dirac) = d.value

entropy(d::Dirac{T}) where {T} = zero(T)

#### Evaluation

pdf(d::Dirac, x::Real) = insupport(d, x) ? 1.0 : 0.0
logpdf(d::Dirac, x::Real) = insupport(d, x) ? 0.0 : -Inf

cdf(d::Dirac, x::Real) = x < d.value ? 0.0 : isnan(x) ? NaN : 1.0
logcdf(d::Dirac, x::Real) = x < d.value ? -Inf : isnan(x) ? NaN : 0.0
ccdf(d::Dirac, x::Real) = x < d.value ? 1.0 : isnan(x) ? NaN : 0.0
logccdf(d::Dirac, x::Real) = x < d.value ? 0.0 : isnan(x) ? NaN : -Inf

quantile(d::Dirac{T}, p::Real) where {T} = 0 <= p <= 1 ? d.value : T(NaN)

mgf(d::Dirac, t) = exp(t * d.value)
cgf(d::Dirac, t) = t*d.value
cf(d::Dirac, t) = cis(t * d.value)

#### Sampling

rand(rng::AbstractRNG, d::Dirac) = d.value

"""
    Arcsine(a,b)

The *Arcsine distribution* has probability density function

```math
f(x) = \\frac{1}{\\pi \\sqrt{(x - a) (b - x)}}, \\quad x \\in [a, b]
```

```julia
Arcsine()        # Arcsine distribution with support [0, 1]
Arcsine(b)       # Arcsine distribution with support [0, b]
Arcsine(a, b)    # Arcsine distribution with support [a, b]

params(d)        # Get the parameters, i.e. (a, b)
minimum(d)       # Get the lower bound, i.e. a
maximum(d)       # Get the upper bound, i.e. b
location(d)      # Get the left bound, i.e. a
scale(d)         # Get the span of the support, i.e. b - a
```

External links

* [Arcsine distribution on Wikipedia](http://en.wikipedia.org/wiki/Arcsine_distribution)

"""
immutable Arcsine{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T

    (::Type{Arcsine{T}}){T}(a::T, b::T) = (@check_args(Arcsine, a < b); new{T}(a, b))
end

Arcsine{T<:Real}(a::T, b::T) = Arcsine{T}(a, b)
Arcsine(a::Real, b::Real) = Arcsine(promote(a, b)...)
Arcsine(a::Integer, b::Integer) = Arcsine(Float64(a), Float64(b))
Arcsine(b::Real) = Arcsine(0.0, b)
Arcsine() = Arcsine(0.0, 1.0)

@distr_support Arcsine d.a d.b

#### Conversions
function convert{T<:Real}(::Type{Arcsine{T}}, a::Real, b::Real)
    Arcsine(T(a), T(b))
end
function convert{T <: Real, S <: Real}(::Type{Arcsine{T}}, d::Arcsine{S})
    Arcsine(T(d.a), T(d.b))
end

### Parameters

location(d::Arcsine) = d.a
scale(d::Arcsine) = d.b - d.a
params(d::Arcsine) = (d.a, d.b)
@inline partype{T<:Real}(d::Arcsine{T}) = T


### Statistics

mean(d::Arcsine) = (d.a + d.b) / 2
median(d::Arcsine) = mean(d)
mode(d::Arcsine) = d.a
modes(d::Arcsine) = [d.a, d.b]

var(d::Arcsine) = abs2(d.b - d.a) / 8
skewness{T<:Real}(d::Arcsine{T}) = zero(T)
kurtosis{T<:Real}(d::Arcsine{T}) = -T(3/2)

entropy(d::Arcsine) = -0.24156447527049044469 + log(scale(d))


### Evaluation

function pdf(d::Arcsine, x::Real)
    insupport(d, x) ? one(d.a) / (π * sqrt((x - d.a) * (d.b - x))) : zero(d.a)
end

function logpdf{T<:Real}(d::Arcsine{T}, x::Real)
    insupport(d, x) ? -(logπ + log((x - d.a) * (d.b - x))/2) : -T(Inf)
end

cdf{T<:Real}(d::Arcsine{T}, x::Real) = x < d.a ? zero(T) :
                              x > d.b ? one(T) :
                              0.636619772367581343 * asin(sqrt((x - d.a) / (d.b - d.a)))

quantile(d::Arcsine, p::Real) = location(d) + abs2(sin(halfπ * p)) * scale(d)


### Sampling

rand(d::Arcsine) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Arcsine) = quantile(d, rand(rng))

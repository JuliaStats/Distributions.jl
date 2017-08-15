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
struct Arcsine{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T

    Arcsine{T}(a::T, b::T) where {T} = (@check_args(Arcsine, a < b); new{T}(a, b))
end

Arcsine(a::T, b::T) where {T<:Real} = Arcsine{T}(a, b)
Arcsine(a::Real, b::Real) = Arcsine(promote(a, b)...)
Arcsine(a::Integer, b::Integer) = Arcsine(Float64(a), Float64(b))
Arcsine(b::Real) = Arcsine(0.0, b)
Arcsine() = Arcsine(0.0, 1.0)

@distr_support Arcsine d.a d.b

#### Conversions
function convert(::Type{Arcsine{T}}, a::Real, b::Real) where T<:Real
    Arcsine(T(a), T(b))
end
function convert(::Type{Arcsine{T}}, d::Arcsine{S}) where {T <: Real, S <: Real}
    Arcsine(T(d.a), T(d.b))
end

### Parameters

location(d::Arcsine) = d.a
scale(d::Arcsine) = d.b - d.a
params(d::Arcsine) = (d.a, d.b)
@inline partype(d::Arcsine{T}) where {T<:Real} = T


### Statistics

mean(d::Arcsine) = (d.a + d.b) / 2
median(d::Arcsine) = mean(d)
mode(d::Arcsine) = d.a
modes(d::Arcsine) = [d.a, d.b]

var(d::Arcsine) = abs2(d.b - d.a) / 8
skewness(d::Arcsine{T}) where {T<:Real} = zero(T)
kurtosis(d::Arcsine{T}) where {T<:Real} = -T(3/2)

entropy(d::Arcsine) = -0.24156447527049044469 + log(scale(d))


### Evaluation

function pdf(d::Arcsine, x::Real)
    insupport(d, x) ? one(d.a) / (π * sqrt((x - d.a) * (d.b - x))) : zero(d.a)
end

function logpdf(d::Arcsine{T}, x::Real) where T<:Real
    insupport(d, x) ? -(logπ + log((x - d.a) * (d.b - x))/2) : -T(Inf)
end

cdf(d::Arcsine{T}, x::Real) where {T<:Real} = x < d.a ? zero(T) :
                              x > d.b ? one(T) :
                              0.636619772367581343 * asin(sqrt((x - d.a) / (d.b - d.a)))

quantile(d::Arcsine, p::Real) = location(d) + abs2(sin(halfπ * p)) * scale(d)


### Sampling

rand(d::Arcsine) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Arcsine) = quantile(d, rand(rng))

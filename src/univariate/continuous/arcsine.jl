"""
    Arcsine <: ContinuousUnivariatDistribution

The *arcsine* probability distribution 

# Constructors

    Arcsine(a=0,b=1)

Construct an `Arcsine` distribution object with minimum `a` and maximum `b`.

# Details
The arcsine distribution has probability density function

```math
f(x) = \\frac{1}{\\pi \\sqrt{(x - a) (b - x)}}, \\quad x \\in [a, b]
```

# Examples
```julia
Arcsine()
Arcsine(b=10)
Arcsine(a=2, b=2)
```

# External links

* [Arcsine distribution on Wikipedia](http://en.wikipedia.org/wiki/Arcsine_distribution)

"""
struct Arcsine{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T

    Arcsine{T}(a::T, b::T) where {T} = (@check_args(Arcsine, a < b); new{T}(a, b))
end

Arcsine(a::T, b::T) where {T<:Real} = Arcsine{T}(a, b)
Arcsine(a::Real, b::Real) = Arcsine(promote(a, b)...)
Arcsine(a::Integer, b::Integer) = Arcsine(float(a), float(b))

@kwdispatch (::Type{D})() where {D<:Arcsine} begin
    () -> D(0,1)
    (a) -> D(a,1)
    (b) -> D(0,b)
    (a,b) -> D(a,b)
end

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

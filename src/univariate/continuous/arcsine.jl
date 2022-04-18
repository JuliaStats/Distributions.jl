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

Use `Arcsine(a, b, check_args=false)` to bypass argument checks.
"""
struct Arcsine{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    Arcsine{T}(a::T, b::T) where {T<:Real} = new{T}(a, b)
end

function Arcsine(a::T, b::T; check_args::Bool=true) where {T <: Real}
    @check_args Arcsine a < b
    return Arcsine{T}(a, b)
end

Arcsine(a::Real, b::Real; check_args::Bool=true) = Arcsine(promote(a, b)...; check_args=check_args)
Arcsine(a::Integer, b::Integer; check_args::Bool=true) = Arcsine(float(a), float(b); check_args=check_args)
Arcsine(b::Real; check_args::Bool=true) = Arcsine(zero(b), b; check_args=check_args)
Arcsine() = Arcsine{Float64}(0.0, 1.0)

@distr_support Arcsine d.a d.b

#### Conversions
function convert(::Type{Arcsine{T}}, a::Real, b::Real) where T<:Real
    Arcsine(T(a), T(b))
end
Base.convert(::Type{Arcsine{T}}, d::Arcsine) where {T<:Real} = Arcsine{T}(T(d.a), T(d.b))
Base.convert(::Type{Arcsine{T}}, d::Arcsine{T}) where {T<:Real} = d

### Parameters

location(d::Arcsine) = d.a
scale(d::Arcsine) = d.b - d.a
params(d::Arcsine) = (d.a, d.b)
partype(::Arcsine{T}) where {T} = T


### Statistics

mean(d::Arcsine) = (d.a + d.b) / 2
median(d::Arcsine) = mean(d)
mode(d::Arcsine) = d.a
modes(d::Arcsine) = [d.a, d.b]

var(d::Arcsine) = abs2(d.b - d.a) / 8
skewness(d::Arcsine{T}) where {T} = zero(T)
kurtosis(d::Arcsine{T}) where {T} = -T(3/2)

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

function gradlogpdf(d::Arcsine{T}, x::R) where {T, R <: Real}
    TP = promote_type(T, R)
    (a, b) = extrema(d)
    # on the bounds, we consider the gradient limit inside the domain
    # right side for the left bound,
    # left side for the right bound
    a < x <= b || return TP(-Inf)
    x == b && return TP(Inf)
    return TP(0.5 * (inv(b - x) - inv(x - a)))
end

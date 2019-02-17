"""
    Exponential <: ContinuousUnivariateDistribution

The *exponential* probability distribution.

# Constructors

    Exponential(θ|theta|scale=1)

Construct an `Exponential` object with scale `θ`.

    Exponential(β|beta|rate=1)

Construct an `Exponential` object with rate `β=1/θ`.

# Details

The exponential distribution with scale parameter `θ` has probability density function

```math
f(x; \\theta) = \\frac{1}{\\theta} e^{-\\frac{x}{\\theta}}, \\quad x > 0
```

# Examples

```julia
Exponential()
Exponential(θ=3)
```

# External links

* [Exponential distribution on Wikipedia](http://en.wikipedia.org/wiki/Exponential_distribution)

"""
struct Exponential{T<:Real} <: ContinuousUnivariateDistribution
    θ::T        # note: scale not rate

    Exponential{T}(θ::Real) where {T} = (@check_args(Exponential, θ > zero(θ)); new{T}(θ))
end

Exponential(θ::T) where {T<:Real} = Exponential{T}(θ)
Exponential(θ::Integer) = Exponential(float(θ))

@kwdispatch (::Type{D})(;theta=>θ, scale=>θ, beta=>β, rate=>β) where {D<:Exponential} begin
    () -> D(1)
    (θ) -> D(θ)
    (β) -> D(inv(β))
end

@distr_support Exponential 0.0 Inf

### Conversions
convert(::Type{Exponential{T}}, θ::S) where {T <: Real, S <: Real} = Exponential(T(θ))
convert(::Type{Exponential{T}}, d::Exponential{S}) where {T <: Real, S <: Real} = Exponential(T(d.θ))


#### Parameters

scale(d::Exponential) = d.θ
rate(d::Exponential) = 1 / d.θ

params(d::Exponential) = (d.θ,)
@inline partype(d::Exponential{T}) where {T<:Real} = T


#### Statistics

mean(d::Exponential) = d.θ
median(d::Exponential) = logtwo * d.θ
mode(d::Exponential{T}) where {T<:Real} = zero(T)

var(d::Exponential) = d.θ^2
skewness(d::Exponential{T}) where {T<:Real} = T(2)
kurtosis(d::Exponential{T}) where {T<:Real} = T(6)

entropy(d::Exponential) = 1 + log(d.θ)


#### Evaluation

zval(d::Exponential, x::Real) = x / d.θ
xval(d::Exponential, z::Real) = z * d.θ

pdf(d::Exponential, x::Real) = (λ = rate(d); x < 0 ? zero(λ) : λ * exp(-λ * x))
function logpdf(d::Exponential{T}, x::Real) where T<:Real
    (λ = rate(d); x < 0 ? -T(Inf) : log(λ) - λ * x)
end

cdf(d::Exponential{T}, x::Real) where {T<:Real} = x > 0 ? -expm1(-zval(d, x)) : zero(T)
ccdf(d::Exponential{T}, x::Real) where {T<:Real} = x > 0 ? exp(-zval(d, x)) : zero(T)
logcdf(d::Exponential{T}, x::Real) where {T<:Real} = x > 0 ? log1mexp(-zval(d, x)) : -T(Inf)
logccdf(d::Exponential{T}, x::Real) where {T<:Real} = x > 0 ? -zval(d, x) : zero(T)

quantile(d::Exponential, p::Real) = -xval(d, log1p(-p))
cquantile(d::Exponential, p::Real) = -xval(d, log(p))
invlogcdf(d::Exponential, lp::Real) = -xval(d, log1mexp(lp))
invlogccdf(d::Exponential, lp::Real) = -xval(d, lp)

gradlogpdf(d::Exponential{T}, x::Real) where {T<:Real} = x > 0 ? -rate(d) : zero(T)

mgf(d::Exponential, t::Real) = 1/(1 - t * scale(d))
cf(d::Exponential, t::Real) = 1/(1 - t * im * scale(d))


#### Sampling
rand(d::Exponential) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Exponential) = xval(d, randexp(rng))


#### Fit model

struct ExponentialStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    sw::Float64   # sum of sample weights

    ExponentialStats(sx::Real, sw::Real) = new(sx, sw)
end

suffstats(::Type{Exponential}, x::AbstractArray{T}) where {T<:Real} = ExponentialStats(sum(x), length(x))
suffstats(::Type{Exponential}, x::AbstractArray{T}, w::AbstractArray{Float64}) where {T<:Real} = ExponentialStats(dot(x, w), sum(w))

fit_mle(::Type{Exponential}, ss::ExponentialStats) = Exponential(ss.sx / ss.sw)

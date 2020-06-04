"""
    Exponential(θ)

The *Exponential distribution* with scale parameter `θ` has probability density function

```math
f(x; \\theta) = \\frac{1}{\\theta} e^{-\\frac{x}{\\theta}}, \\quad x > 0
```

```julia
Exponential()      # Exponential distribution with unit scale, i.e. Exponential(1)
Exponential(b)     # Exponential distribution with scale b

params(d)          # Get the parameters, i.e. (b,)
scale(d)           # Get the scale parameter, i.e. b
rate(d)            # Get the rate parameter, i.e. 1 / b
```

External links

* [Exponential distribution on Wikipedia](http://en.wikipedia.org/wiki/Exponential_distribution)

"""
struct Exponential{T<:Real} <: ContinuousUnivariateDistribution
    θ::T        # note: scale not rate
    Exponential{T}(θ::T) where {T} = new{T}(θ)
end

function Exponential(θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Exponential, θ > zero(θ))
    return Exponential{T}(θ)
end

Exponential(θ::Integer) = Exponential(float(θ))
Exponential() = Exponential(1.0, check_args=false)

@distr_support Exponential 0.0 Inf

### Conversions
convert(::Type{Exponential{T}}, θ::S) where {T <: Real, S <: Real} = Exponential(T(θ))
convert(::Type{Exponential{T}}, d::Exponential{S}) where {T <: Real, S <: Real} = Exponential(T(d.θ), check_args=false)

#### Parameters

scale(d::Exponential) = d.θ
rate(d::Exponential) = inv(d.θ)

params(d::Exponential) = (d.θ,)
partype(::Exponential{T}) where {T<:Real} = T

#### Statistics

mean(d::Exponential) = d.θ
median(d::Exponential) = logtwo * d.θ
mode(::Exponential{T}) where {T<:Real} = zero(T)

var(d::Exponential) = d.θ^2
skewness(::Exponential{T}) where {T} = T(2)
kurtosis(::Exponential{T}) where {T} = T(6)

entropy(d::Exponential{T}) where {T} = one(T) + log(d.θ)

#### Evaluation

zval(d::Exponential, x::Real) = x / d.θ
xval(d::Exponential, z::Real) = z * d.θ

pdf(d::Exponential, x::Real) = (λ = rate(d); x < 0 ? zero(λ) : λ * exp(-λ * x))
function logpdf(d::Exponential{T}, x::Real) where T<:Real
    λ = rate(d)
    x < 0 ? -T(Inf) : log(λ) - λ * x
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
rand(rng::AbstractRNG, d::Exponential) = xval(d, randexp(rng))


#### Fit model

struct ExponentialStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    sw::Float64   # sum of sample weights

    ExponentialStats(sx::Real, sw::Real) = new(sx, sw)
end

suffstats(::Type{<:Exponential}, x::AbstractArray{T}) where {T<:Real} = ExponentialStats(sum(x), length(x))
suffstats(::Type{<:Exponential}, x::AbstractArray{T}, w::AbstractArray{Float64}) where {T<:Real} = ExponentialStats(dot(x, w), sum(w))

fit_mle(::Type{<:Exponential}, ss::ExponentialStats) = Exponential(ss.sx / ss.sw)

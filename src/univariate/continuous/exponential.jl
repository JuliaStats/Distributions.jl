doc"""
    Exponential(θ)

The *Exponential distribution* with scale parameter `θ` has probability density function

$f(x; \theta) = \frac{1}{\theta} e^{-\frac{x}{\theta}}, \quad x > 0$

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
immutable Exponential{T<:Real} <: ContinuousUnivariateDistribution
    θ::T		# note: scale not rate

    (::Type{Exponential{T}}){T}(θ::Real) = (@check_args(Exponential, θ > zero(θ)); new{T}(θ))
end

Exponential{T<:Real}(θ::T) = Exponential{T}(θ)
Exponential(θ::Integer) = Exponential(Float64(θ))
Exponential() = Exponential(1.0)

@distr_support Exponential 0.0 Inf

### Conversions
convert{T <: Real, S <: Real}(::Type{Exponential{T}}, θ::S) = Exponential(T(θ))
convert{T <: Real, S <: Real}(::Type{Exponential{T}}, d::Exponential{S}) = Exponential(T(d.θ))


#### Parameters

scale(d::Exponential) = d.θ
rate(d::Exponential) = 1 / d.θ

params(d::Exponential) = (d.θ,)
@inline partype{T<:Real}(d::Exponential{T}) = T


#### Statistics

mean(d::Exponential) = d.θ
median(d::Exponential) = logtwo * d.θ
mode{T<:Real}(d::Exponential{T}) = zero(T)

var(d::Exponential) = d.θ^2
skewness{T<:Real}(d::Exponential{T}) = T(2)
kurtosis{T<:Real}(d::Exponential{T}) = T(6)

entropy(d::Exponential) = 1 + log(d.θ)


#### Evaluation

zval(d::Exponential, x::Real) = x / d.θ
xval(d::Exponential, z::Real) = z * d.θ

pdf(d::Exponential, x::Real) = (λ = rate(d); x < 0 ? zero(λ) : λ * exp(-λ * x))
function logpdf{T<:Real}(d::Exponential{T}, x::Real)
    (λ = rate(d); x < 0 ? -T(Inf) : log(λ) - λ * x)
end

cdf{T<:Real}(d::Exponential{T}, x::Real) = x > 0 ? -expm1(-zval(d, x)) : zero(T)
ccdf{T<:Real}(d::Exponential{T}, x::Real) = x > 0 ? exp(-zval(d, x)) : zero(T)
logcdf{T<:Real}(d::Exponential{T}, x::Real) = x > 0 ? log1mexp(-zval(d, x)) : -T(Inf)
logccdf{T<:Real}(d::Exponential{T}, x::Real) = x > 0 ? -zval(d, x) : zero(T)

quantile(d::Exponential, p::Real) = -xval(d, log1p(-p))
cquantile(d::Exponential, p::Real) = -xval(d, log(p))
invlogcdf(d::Exponential, lp::Real) = -xval(d, log1mexp(lp))
invlogccdf(d::Exponential, lp::Real) = -xval(d, lp)

gradlogpdf{T<:Real}(d::Exponential{T}, x::Real) = x > 0 ? -rate(d) : zero(T)

mgf(d::Exponential, t::Real) = 1/(1 - t * scale(d))
cf(d::Exponential, t::Real) = 1/(1 - t * im * scale(d))


#### Sampling
rand(d::Exponential) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Exponential) = xval(d, randexp(rng))


#### Fit model

immutable ExponentialStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    sw::Float64   # sum of sample weights

    ExponentialStats(sx::Real, sw::Real) = new(sx, sw)
end

suffstats{T<:Real}(::Type{Exponential}, x::AbstractArray{T}) = ExponentialStats(sum(x), length(x))
suffstats{T<:Real}(::Type{Exponential}, x::AbstractArray{T}, w::AbstractArray{Float64}) = ExponentialStats(dot(x, w), sum(w))

fit_mle(::Type{Exponential}, ss::ExponentialStats) = Exponential(ss.sx / ss.sw)

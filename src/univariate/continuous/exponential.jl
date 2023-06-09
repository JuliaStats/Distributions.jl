"""
    Exponential(θ)

The *Exponential distribution* with scale parameter `θ` has probability density function

```math
f(x; \\theta) = \\frac{1}{\\theta} e^{-\\frac{x}{\\theta}}, \\quad x > 0
```

```julia
Exponential()      # Exponential distribution with unit scale, i.e. Exponential(1)
Exponential(θ)     # Exponential distribution with scale θ

params(d)          # Get the parameters, i.e. (θ,)
scale(d)           # Get the scale parameter, i.e. θ
rate(d)            # Get the rate parameter, i.e. 1 / θ
```

External links

* [Exponential distribution on Wikipedia](http://en.wikipedia.org/wiki/Exponential_distribution)

"""
struct Exponential{T<:Real} <: ContinuousUnivariateDistribution
    θ::T        # note: scale not rate
    Exponential{T}(θ::T) where {T} = new{T}(θ)
end

function Exponential(θ::Real; check_args::Bool=true)
    @check_args Exponential (θ, θ > zero(θ))
    return Exponential{typeof(θ)}(θ)
end

Exponential(θ::Integer; check_args::Bool=true) = Exponential(float(θ); check_args=check_args)
Exponential() = Exponential{Float64}(1.0)

@distr_support Exponential 0.0 Inf

### Conversions
convert(::Type{Exponential{T}}, θ::S) where {T <: Real, S <: Real} = Exponential(T(θ))
function Base.convert(::Type{Exponential{T}}, d::Exponential) where {T<:Real}
    return Exponential(T(d.θ))
end
Base.convert(::Type{Exponential{T}}, d::Exponential{T}) where {T<:Real} = d

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

function kldivergence(p::Exponential, q::Exponential)
    λq_over_λp = scale(q) / scale(p)
    return -logmxp1(λq_over_λp)
end

#### Evaluation

zval(d::Exponential, x::Real) = max(x / d.θ, 0)
xval(d::Exponential, z::Real) = z * d.θ

function pdf(d::Exponential, x::Real)
    λ = rate(d)
    z = λ * exp(-λ * max(x, 0))
    return x < 0 ? zero(z) : z
end
function logpdf(d::Exponential, x::Real)
    λ = rate(d)
    z = log(λ) - λ * x
    return x < 0 ? oftype(z, -Inf) : z
end

cdf(d::Exponential, x::Real) = -expm1(-zval(d, x))
ccdf(d::Exponential, x::Real) = exp(-zval(d, x))
logcdf(d::Exponential, x::Real) = log1mexp(-zval(d, x))
logccdf(d::Exponential, x::Real) = -zval(d, x)

quantile(d::Exponential, p::Real) = -xval(d, log1p(-p))
cquantile(d::Exponential, p::Real) = -xval(d, log(p))
invlogcdf(d::Exponential, lp::Real) = -xval(d, log1mexp(lp))
invlogccdf(d::Exponential, lp::Real) = -xval(d, lp)

gradlogpdf(d::Exponential{T}, x::Real) where {T<:Real} = x > 0 ? -rate(d) : zero(T)

mgf(d::Exponential, t::Real) = 1/(1 - t * scale(d))
function cgf(d::Exponential, t)
    μ = mean(d)
    return - log1p(- t * μ)
end
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

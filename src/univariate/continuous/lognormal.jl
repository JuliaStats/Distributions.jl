"""
    LogNormal(μ,σ)

The *log normal distribution* is the distribution of the exponential of a [`Normal`](@ref) variate: if ``X \\sim \\operatorname{Normal}(\\mu, \\sigma)`` then
``\\exp(X) \\sim \\operatorname{LogNormal}(\\mu,\\sigma)``. The probability density function is

```math
f(x; \\mu, \\sigma) = \\frac{1}{x \\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(\\log(x) - \\mu)^2}{2 \\sigma^2} \\right),
\\quad x > 0
```

```julia
LogNormal()          # Log-normal distribution with zero log-mean and unit scale
LogNormal(mu)        # Log-normal distribution with log-mean mu and unit scale
LogNormal(mu, sig)   # Log-normal distribution with log-mean mu and scale sig

params(d)            # Get the parameters, i.e. (mu, sig)
meanlogx(d)          # Get the mean of log(X), i.e. mu
varlogx(d)           # Get the variance of log(X), i.e. sig^2
stdlogx(d)           # Get the standard deviation of log(X), i.e. sig
```

External links

* [Log normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Log-normal_distribution)

"""
struct LogNormal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    LogNormal{T}(μ::T, σ::T) where {T} = (@check_args(LogNormal, σ > zero(σ)); new{T}(μ, σ))
end

LogNormal(μ::T, σ::T) where {T<:Real} = LogNormal{T}(μ, σ)
LogNormal(μ::Real, σ::Real) = LogNormal(promote(μ, σ)...)
LogNormal(μ::Integer, σ::Integer) = LogNormal(Float64(μ), Float64(σ))
LogNormal(μ::Real) = LogNormal(μ, 1.0)
LogNormal() = LogNormal(0.0, 1.0)

@distr_support LogNormal 0.0 Inf

#### Conversions
convert(::Type{LogNormal{T}}, μ::S, σ::S) where {T <: Real, S <: Real} = LogNormal(T(μ), T(σ))
convert(::Type{LogNormal{T}}, d::LogNormal{S}) where {T <: Real, S <: Real} = LogNormal(T(d.μ), T(d.σ))

#### Parameters

params(d::LogNormal) = (d.μ, d.σ)
@inline partype(d::LogNormal{T}) where {T<:Real} = T

#### Statistics

meanlogx(d::LogNormal) = d.μ
varlogx(d::LogNormal) = abs2(d.σ)
stdlogx(d::LogNormal) = d.σ

mean(d::LogNormal) = ((μ, σ) = params(d); exp(μ + σ^2/2))
median(d::LogNormal) = exp(d.μ)
mode(d::LogNormal) = ((μ, σ) = params(d); exp(μ - σ^2))

function var(d::LogNormal)
    (μ, σ) = params(d)
    σ2 = σ^2
    (exp(σ2) - 1) * exp(2μ + σ2)
end

function skewness(d::LogNormal)
    σ2 = varlogx(d)
    e = exp(σ2)
    (e + 2) * sqrt(e - 1)
end

function kurtosis(d::LogNormal)
    σ2 = varlogx(d)
    e = exp(σ2)
    e2 = e * e
    e3 = e2 * e
    e4 = e3 * e
    e4 + 2*e3 + 3*e2 - 6
end

function entropy(d::LogNormal)
    (μ, σ) = params(d)
    (1 + log(twoπ * σ^2))/2 + μ
end


#### Evalution

pdf(d::LogNormal, x::Real) = normpdf(d.μ, d.σ, log(x)) / x
function logpdf(d::LogNormal{T}, x::Real) where T<:Real
    if !insupport(d, x)
        return -T(Inf)
    else
        lx = log(x)
        return normlogpdf(d.μ, d.σ, lx) - lx
    end
end

cdf(d::LogNormal{T}, x::Real) where {T<:Real} = x > 0 ? normcdf(d.μ, d.σ, log(x)) : zero(T)
ccdf(d::LogNormal{T}, x::Real) where {T<:Real} = x > 0 ? normccdf(d.μ, d.σ, log(x)) : one(T)
logcdf(d::LogNormal{T}, x::Real) where {T<:Real} = x > 0 ? normlogcdf(d.μ, d.σ, log(x)) : -T(Inf)
logccdf(d::LogNormal{T}, x::Real) where {T<:Real} = x > 0 ? normlogccdf(d.μ, d.σ, log(x)) : zero(T)

quantile(d::LogNormal, q::Real) = exp(norminvcdf(d.μ, d.σ, q))
cquantile(d::LogNormal, q::Real) = exp(norminvccdf(d.μ, d.σ, q))
invlogcdf(d::LogNormal, lq::Real) = exp(norminvlogcdf(d.μ, d.σ, lq))
invlogccdf(d::LogNormal, lq::Real) = exp(norminvlogccdf(d.μ, d.σ, lq))

function gradlogpdf(d::LogNormal{T}, x::Real) where T<:Real
    (μ, σ) = params(d)
    x > 0 ? - ((log(x) - μ) / (σ^2) + 1) / x : zero(T)
end

# mgf(d::LogNormal)
# cf(d::LogNormal)


#### Sampling

rand(d::LogNormal) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::LogNormal) = exp(randn(rng) * d.σ + d.μ)

## Fitting

function fit_mle(::Type{LogNormal}, x::AbstractArray{T}) where T<:Real
    lx = log.(x)
    μ, σ = mean_and_std(lx)
    LogNormal(μ, σ)
end

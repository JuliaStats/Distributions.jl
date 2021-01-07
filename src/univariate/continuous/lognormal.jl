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
    LogNormal{T}(μ::T, σ::T) where {T} = new{T}(μ, σ)
end

function LogNormal(μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(LogNormal, σ ≥ zero(σ))
    return LogNormal{T}(μ, σ)
end

LogNormal(μ::Real, σ::Real) = LogNormal(promote(μ, σ)...)
LogNormal(μ::Integer, σ::Integer) = LogNormal(float(μ), float(σ))
LogNormal(μ::T) where {T <: Real} = LogNormal(μ, one(T))
LogNormal() = LogNormal(0.0, 1.0, check_args=false)

@distr_support LogNormal 0.0 Inf

#### Conversions
convert(::Type{LogNormal{T}}, μ::S, σ::S) where {T <: Real, S <: Real} = LogNormal(T(μ), T(σ))
convert(::Type{LogNormal{T}}, d::LogNormal{S}) where {T <: Real, S <: Real} = LogNormal(T(d.μ), T(d.σ), check_args=false)

#### Parameters

params(d::LogNormal) = (d.μ, d.σ)
partype(::LogNormal{T}) where {T} = T

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

function pdf(d::LogNormal, x::Real)
    if x ≤ zero(x)
        logx = log(zero(x))
        x = one(x)
    else
        logx = log(x)
    end
    return pdf(Normal(d.μ, d.σ), logx) / x
end

function logpdf(d::LogNormal, x::Real)
    if x ≤ zero(x)
        logx = log(zero(x))
        b = zero(logx)
    else
        logx = log(x)
        b = logx
    end
    return logpdf(Normal(d.μ, d.σ), logx) - b
end

function cdf(d::LogNormal, x::Real)
    logx = x ≤ zero(x) ? log(zero(x)) : log(x)
    return cdf(Normal(d.μ, d.σ), logx)
end

function ccdf(d::LogNormal, x::Real)
    logx = x ≤ zero(x) ? log(zero(x)) : log(x)
    return ccdf(Normal(d.μ, d.σ), logx)
    end

function logcdf(d::LogNormal, x::Real)
    logx = x ≤ zero(x) ? log(zero(x)) : log(x)
    return logcdf(Normal(d.μ, d.σ), logx)
end

function logccdf(d::LogNormal, x::Real)
    logx = x ≤ zero(x) ? log(zero(x)) : log(x)
    return logccdf(Normal(d.μ, d.σ), logx)
end

quantile(d::LogNormal, q::Real) = exp(quantile(Normal(params(d)...), q))
cquantile(d::LogNormal, q::Real) = exp(cquantile(Normal(params(d)...), q))
invlogcdf(d::LogNormal, lq::Real) = exp(invlogcdf(Normal(params(d)...), lq))
invlogccdf(d::LogNormal, lq::Real) = exp(invlogccdf(Normal(params(d)...), lq))

function gradlogpdf(d::LogNormal, x::Real)
    outofsupport = x ≤ zero(x)
    y = outofsupport ? one(x) : x
    z = (gradlogpdf(Normal(d.μ, d.σ), log(y)) - 1) / y
    return outofsupport ? zero(z) : z
end

# mgf(d::LogNormal)
# cf(d::LogNormal)


#### Sampling

rand(rng::AbstractRNG, d::LogNormal) = exp(randn(rng) * d.σ + d.μ)

## Fitting

function fit_mle(::Type{<:LogNormal}, x::AbstractArray{T}) where T<:Real
    lx = log.(x)
    μ, σ = mean_and_std(lx)
    LogNormal(μ, σ)
end

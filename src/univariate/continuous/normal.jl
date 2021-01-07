"""
    Normal(μ,σ)

The *Normal distribution* with mean `μ` and standard deviation `σ≥0` has probability density function

```math
f(x; \\mu, \\sigma) = \\frac{1}{\\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(x - \\mu)^2}{2 \\sigma^2} \\right)
```

Note that if `σ == 0`, then the distribution is a point mass concentrated at `μ`.
Though not technically a continuous distribution, it is allowed so as to account for cases where `σ` may have underflowed,
and the functions are defined by taking the pointwise limit as ``σ → 0``.

```julia
Normal()          # standard Normal distribution with zero mean and unit variance
Normal(mu)        # Normal distribution with mean mu and unit variance
Normal(mu, sig)   # Normal distribution with mean mu and variance sig^2

params(d)         # Get the parameters, i.e. (mu, sig)
mean(d)           # Get the mean, i.e. mu
std(d)            # Get the standard deviation, i.e. sig
```

External links

* [Normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Normal_distribution)

"""
struct Normal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    Normal{T}(µ::T, σ::T) where {T<:Real} = new{T}(µ, σ)
end

function Normal(μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Normal, σ >= zero(σ))
    return Normal{T}(μ, σ)
end

#### Outer constructors
Normal(μ::Real, σ::Real) = Normal(promote(μ, σ)...)
Normal(μ::Integer, σ::Integer) = Normal(float(μ), float(σ))
Normal(μ::T) where {T <: Real} = Normal(μ, one(T))
Normal() = Normal(0.0, 1.0, check_args=false)

const Gaussian = Normal

# #### Conversions
convert(::Type{Normal{T}}, μ::S, σ::S) where {T <: Real, S <: Real} = Normal(T(μ), T(σ))
convert(::Type{Normal{T}}, d::Normal{S}) where {T <: Real, S <: Real} = Normal(T(d.μ), T(d.σ), check_args=false)

@distr_support Normal -Inf Inf

#### Parameters

params(d::Normal) = (d.μ, d.σ)
@inline partype(d::Normal{T}) where {T<:Real} = T

location(d::Normal) = d.μ
scale(d::Normal) = d.σ

Base.eltype(::Type{Normal{T}}) where {T} = T

#### Statistics

mean(d::Normal) = d.μ
median(d::Normal) = d.μ
mode(d::Normal) = d.μ

var(d::Normal) = abs2(d.σ)
std(d::Normal) = d.σ
skewness(d::Normal{T}) where {T<:Real} = zero(T)
kurtosis(d::Normal{T}) where {T<:Real} = zero(T)

entropy(d::Normal) = (log2π + 1)/2 + log(d.σ)

#### Evaluation

# Helpers
"""
    xval(d::Normal, z::Real)

Computes the x-value based on a Normal distribution and a z-value.
"""
function xval(d::Normal, z::Real)
    if isinf(z) && iszero(d.σ)
        d.μ + one(d.σ) * z
    else
        d.μ + d.σ * z
    end
end
"""
    zval(d::Normal, x::Real)

Computes the z-value based on a Normal distribution and a x-value.
"""
zval(d::Normal, x::Real) = (x - d.μ) / d.σ

gradlogpdf(d::Normal, x::Real) = -zval(d, x) / d.σ

# logpdf
_normlogpdf(z::Real) = -(abs2(z) + log2π)/2

function logpdf(d::Normal, x::Real)
    μ, σ = d.μ, d.σ
    if iszero(d.σ)
        if x == μ
            z = zval(Normal(μ, one(σ)), x)
        else
            z = zval(d, x)
            σ = one(σ)
        end
    else
        z = zval(Normal(μ, σ), x)
    end
    return _normlogpdf(z) - log(σ)
end

# pdf
_normpdf(z::Real) = exp(-abs2(z)/2) * invsqrt2π

function pdf(d::Normal, x::Real)
    μ, σ = d.μ, d.σ
    if iszero(σ)
        if x == μ
            z = zval(Normal(μ, one(σ)), x)
        else
            z = zval(d, x)
            σ = one(σ)
        end
    else
        z = zval(Normal(μ, σ), x)
    end
    return _normpdf(z) / σ
end

# logcdf
function _normlogcdf(z::Real)
    if z < -one(z)
        return log(erfcx(-z * invsqrt2)/2) - abs2(z)/2
    else
        return log1p(-erfc(z * invsqrt2)/2)
    end
end

function logcdf(d::Normal, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(Normal(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    return _normlogcdf(z)
end

# logccdf
function _normlogccdf(z::Real)
    if z > one(z)
        return log(erfcx(z * invsqrt2)/2) - abs2(z)/2
    else
        return log1p(-erfc(-z * invsqrt2)/2)
    end
end

function logccdf(d::Normal, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(Normal(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    return _normlogccdf(z)
end

# cdf
_normcdf(z::Real) = erfc(-z * invsqrt2)/2

function cdf(d::Normal, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(Normal(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    return _normcdf(z)
end

# ccdf
_normccdf(z::Real) = erfc(z * invsqrt2)/2

function ccdf(d::Normal, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(Normal(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    return _normccdf(z)
end

# quantile
function quantile(d::Normal, p::Real)
    # Promote to ensure that we don't compute erfcinv in low precision and then promote
    _p, _μ, _σ = promote(float(p), d.μ, d.σ)
    q = xval(d, -erfcinv(2*_p) * sqrt2)
    if isnan(_p)
        return oftype(q, _p)
    elseif iszero(_σ)
        # Quantile not uniquely defined at p=0 and p=1 when σ=0
        if iszero(_p)
            return oftype(q, -Inf)
        elseif isone(_p)
            return oftype(q, Inf)
        else
            return oftype(q, _μ)
        end
    end
    return q
end

# cquantile
function cquantile(d::Normal, p::Real)
    # Promote to ensure that we don't compute erfcinv in low precision and then promote
    _p, _μ, _σ = promote(float(p), d.μ, d.σ)
    q = xval(d, erfcinv(2*_p) * sqrt2)
    if isnan(_p)
        return oftype(q, _p)
    elseif iszero(d.σ)
        # Quantile not uniquely defined at p=0 and p=1 when σ=0
        if iszero(_p)
            return oftype(q, Inf)
        elseif isone(_p)
            return oftype(q, -Inf)
        else
            return oftype(q, _μ)
        end
    end
    return q
end

mgf(d::Normal, t::Real) = exp(t * d.μ + d.σ^2 / 2 * t^2)
cf(d::Normal, t::Real) = exp(im * t * d.μ - d.σ^2 / 2 * t^2)

#### Sampling

rand(rng::AbstractRNG, d::Normal{T}) where {T} = d.μ + d.σ * randn(rng, T)

#### Fitting

struct NormalStats <: SufficientStats
    s::Float64    # (weighted) sum of x
    m::Float64    # (weighted) mean of x
    s2::Float64   # (weighted) sum of (x - μ)^2
    tw::Float64    # total sample weight
end

function suffstats(::Type{<:Normal}, x::AbstractArray{T}) where T<:Real
    n = length(x)

    # compute s
    s = x[1]
    for i = 2:n
        @inbounds s += x[i]
    end
    m = s / n

    # compute s2
    s2 = abs2(x[1] - m)
    for i = 2:n
        @inbounds s2 += abs2(x[i] - m)
    end

    NormalStats(s, m, s2, n)
end

function suffstats(::Type{<:Normal}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    n = length(x)

    # compute s
    tw = w[1]
    s = w[1] * x[1]
    for i = 2:n
        @inbounds wi = w[i]
        @inbounds s += wi * x[i]
        tw += wi
    end
    m = s / tw

    # compute s2
    s2 = w[1] * abs2(x[1] - m)
    for i = 2:n
        @inbounds s2 += w[i] * abs2(x[i] - m)
    end

    NormalStats(s, m, s2, tw)
end

# Cases where μ or σ is known

struct NormalKnownMu <: IncompleteDistribution
    μ::Float64
end

struct NormalKnownMuStats <: SufficientStats
    μ::Float64      # known mean
    s2::Float64     # (weighted) sum of (x - μ)^2
    tw::Float64     # total sample weight
end

function suffstats(g::NormalKnownMu, x::AbstractArray{T}) where T<:Real
    μ = g.μ
    s2 = abs2(x[1] - μ)
    for i = 2:length(x)
        @inbounds s2 += abs2(x[i] - μ)
    end
    NormalKnownMuStats(g.μ, s2, length(x))
end

function suffstats(g::NormalKnownMu, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    μ = g.μ
    s2 = abs2(x[1] - μ) * w[1]
    tw = w[1]
    for i = 2:length(x)
        @inbounds wi = w[i]
        @inbounds s2 += abs2(x[i] - μ) * wi
        tw += wi
    end
    NormalKnownMuStats(g.μ, s2, tw)
end

struct NormalKnownSigma <: IncompleteDistribution
    σ::Float64

    function NormalKnownSigma(σ::Float64)
        σ > 0 || throw(ArgumentError("σ must be a positive value."))
        new(σ)
    end
end

struct NormalKnownSigmaStats <: SufficientStats
    σ::Float64      # known std.dev
    sx::Float64      # (weighted) sum of x
    tw::Float64     # total sample weight
end

function suffstats(g::NormalKnownSigma, x::AbstractArray{T}) where T<:Real
    NormalKnownSigmaStats(g.σ, sum(x), Float64(length(x)))
end

function suffstats(g::NormalKnownSigma, x::AbstractArray{T}, w::AbstractArray{T}) where T<:Real
    NormalKnownSigmaStats(g.σ, dot(x, w), sum(w))
end

# fit_mle based on sufficient statistics

fit_mle(::Type{<:Normal}, ss::NormalStats) = Normal(ss.m, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownMu, ss::NormalKnownMuStats) = Normal(g.μ, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownSigma, ss::NormalKnownSigmaStats) = Normal(ss.sx / ss.tw, g.σ)

# generic fit_mle methods

function fit_mle(::Type{<:Normal}, x::AbstractArray{T}; mu::Float64=NaN, sigma::Float64=NaN) where T<:Real
    if isnan(mu)
        if isnan(sigma)
            fit_mle(Normal, suffstats(Normal, x))
        else
            g = NormalKnownSigma(sigma)
            fit_mle(g, suffstats(g, x))
        end
    else
        if isnan(sigma)
            g = NormalKnownMu(mu)
            fit_mle(g, suffstats(g, x))
        else
            Normal(mu, sigma)
        end
    end
end

function fit_mle(::Type{<:Normal}, x::AbstractArray{T}, w::AbstractArray{Float64}; mu::Float64=NaN, sigma::Float64=NaN) where T<:Real
    if isnan(mu)
        if isnan(sigma)
            fit_mle(Normal, suffstats(Normal, x, w))
        else
            g = NormalKnownSigma(sigma)
            fit_mle(g, suffstats(g, x, w))
        end
    else
        if isnan(sigma)
            g = NormalKnownMu(mu)
            fit_mle(g, suffstats(g, x, w))
        else
            Normal(mu, sigma)
        end
    end
end

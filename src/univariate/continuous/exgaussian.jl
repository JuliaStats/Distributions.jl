"""
    Exgaussian(μ,σ,τ)

The *Exgaussian distribution* is the sum of a normal with mean `μ` and standard deviation `σ>0`, plus an independent exponential with mean `τ>0`.  It has probability density function

```math
f(x; \\mu, \\sigma, \\tau) = NEWJEFF
```

Note that unlike normal.jl, we require σ > 0.

```julia
Exgaussian(μ, σ, τ)   # Exgaussian distribution with normal mean μ, normal variance σ^2, and exponential mean τ

params(d)         # Get the parameters, i.e. (μ, σ, τ)
mean(d)           # Get the mean, i.e. μ+τ
std(d)            # Get the standard deviation, i.e. sqrt(σ^2+τ^2)
```

External links

* [Exgaussian distribution on Wikipedia](http://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)

"""
struct Exgaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    τ::T
    Exgaussian{T}(µ::T, σ::T, τ::T) where {T<:Real} = new{T}(µ, σ, τ)
end

function Exgaussian(μ::T, σ::T, τ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Exgaussian, σ > zero(σ) && τ > zero(τ))
    return Exgaussian{T}(μ, σ, τ)
end

#### Outer constructors
Exgaussian(μ::Real, σ::Real, τ::Real) = Exgaussian(promote(μ, σ, τ)...)
Exgaussian(μ::Integer, σ::Integer, τ::Real) = Exgaussian(float(μ), float(σ), float(τ))

# #### Conversions
convert(::Type{Exgaussian{T}}, μ::S, σ::S, τ::S) where {T <: Real, S <: Real} = Exgaussian(T(μ), T(σ), T(τ))
convert(::Type{Exgaussian{T}}, d::Exgaussian{S}) where {T <: Real, S <: Real} = Exgaussian(T(d.μ), T(d.σ), T(d.τ), check_args=false)

@distr_support Exgaussian -Inf Inf

#### Parameters

params(d::Exgaussian) = (d.μ, d.σ, d.τ)
@inline partype(d::Exgaussian{T}) where {T<:Real} = T

location(d::Exgaussian) = d.μ
scale(d::Exgaussian) = d.σ

Base.eltype(::Type{Exgaussian{T}}) where {T} = T

#### Statistics

mean(d::Exgaussian) = d.μ+d.τ
#median(d::Exgaussian) = d.μ
#mode(d::Exgaussian) = d.μ

#var(d::Exgaussian) = abs2(d.σ)
#std(d::Exgaussian) = d.σ
#skewness(d::Exgaussian{T}) where {T<:Real} = zero(T)
#kurtosis(d::Exgaussian{T}) where {T<:Real} = zero(T)

#entropy(d::Exgaussian) = (log2π + 1)/2 + log(d.σ)

#function kldivergence(p::Exgaussian, q::Exgaussian)
#    μp = mean(p)
#    σ²p = var(p)
#    μq = mean(q)
#    σ²q = var(q)
#    σ²p_over_σ²q = σ²p / σ²q
#    return (abs2(μp - μq) / σ²q - logmxp1(σ²p_over_σ²q)) / 2
#end

#### Evaluation

# Helpers
"""
    xval(d::Exgaussian, z::Real)

Computes the x-value based on a Exgaussian distribution and a z-value.
"""
function xval(d::Exgaussian, z::Real)
    if isinf(z) && iszero(d.σ)
        d.μ + one(d.σ) * z
    else
        d.μ + d.σ * z
    end
end
"""
    zval(d::Exgaussian, x::Real)

Computes the z-value based on a Exgaussian distribution and a x-value.
"""
zval(d::Exgaussian, x::Real) = (x - d.μ) / d.σ

gradlogpdf(d::Exgaussian, x::Real) = -zval(d, x) / d.σ

# logpdf
_normlogpdf(z::Real) = -(abs2(z) + log2π)/2

function logpdf(d::Exgaussian, x::Real)
    μ, σ = d.μ, d.σ
    if iszero(d.σ)
        if x == μ
            z = zval(Exgaussian(μ, one(σ)), x)
        else
            z = zval(d, x)
            σ = one(σ)
        end
    else
        z = zval(Exgaussian(μ, σ), x)
    end
    return _normlogpdf(z) - log(σ)
end

# pdf
_normpdf(z::Real) = exp(-abs2(z)/2) * invsqrt2π

function pdf(d::Exgaussian, x::Real)
    μ, σ, τ = d.μ, d.σ, d.τ
    rate = 1/τ
    # if iszero(σ)
    #     if x == μ
    #         z = zval(Exgaussian(μ, one(σ)), x)
    #     else
    #         z = zval(d, x)
    #         σ = one(σ)
    #     end
    # else
    #     z = zval(Exgaussian(μ, σ), x)
    # end
    # return _normpdf(z) / σ
    t1 = -x*rate + μ*rate + 0.5*(σ*rate)^2
    t2 = (x - μ - σ^2*rate) / σ
    return rate*exp( t1 + _normlogcdf(t2) )
end

# logcdf
function _normlogcdf(z::Real)
    if z < -one(z)
        return log(erfcx(-z * invsqrt2)/2) - abs2(z)/2
    else
        return log1p(-erfc(z * invsqrt2)/2)
    end
end

function logcdf(d::Exgaussian, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(Exgaussian(zero(d.μ), d.σ), one(x))
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

function logccdf(d::Exgaussian, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(Exgaussian(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    return _normlogccdf(z)
end

# cdf
_normcdf(z::Real) = erfc(-z * invsqrt2)/2

function cdf(d::Exgaussian, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(Exgaussian(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    return _normcdf(z)
end

# ccdf
_normccdf(z::Real) = erfc(z * invsqrt2)/2

function ccdf(d::Exgaussian, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(Exgaussian(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    return _normccdf(z)
end

# quantile
function quantile(d::Exgaussian, p::Real)
    # Promote to ensure that we don't compute erfcinv in low precision and then promote
    _p, _μ, _σ = map(float, promote(p, d.μ, d.σ))
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
function cquantile(d::Exgaussian, p::Real)
    # Promote to ensure that we don't compute erfcinv in low precision and then promote
    _p, _μ, _σ = map(float, promote(p, d.μ, d.σ))
    q = xval(d, erfcinv(2*_p) * sqrt2)
    if isnan(_p)
        return oftype(q, _p)
    elseif iszero(_σ)
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

mgf(d::Exgaussian, t::Real) = exp(t * d.μ + d.σ^2 / 2 * t^2)
cf(d::Exgaussian, t::Real) = exp(im * t * d.μ - d.σ^2 / 2 * t^2)

#### Sampling

rand(rng::AbstractRNG, d::Exgaussian{T}) where {T} = d.μ + d.σ * randn(rng, float(T))

#### Fitting

struct ExgaussianStats <: SufficientStats
    s::Float64    # (weighted) sum of x
    m::Float64    # (weighted) mean of x
    s2::Float64   # (weighted) sum of (x - μ)^2
    tw::Float64    # total sample weight
end

function suffstats(::Type{<:Exgaussian}, x::AbstractArray{T}) where T<:Real
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

    ExgaussianStats(s, m, s2, n)
end

function suffstats(::Type{<:Exgaussian}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
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

    ExgaussianStats(s, m, s2, tw)
end

# Cases where μ or σ is known

struct ExgaussianKnownMu <: IncompleteDistribution
    μ::Float64
end

struct ExgaussianKnownMuStats <: SufficientStats
    μ::Float64      # known mean
    s2::Float64     # (weighted) sum of (x - μ)^2
    tw::Float64     # total sample weight
end

function suffstats(g::ExgaussianKnownMu, x::AbstractArray{T}) where T<:Real
    μ = g.μ
    s2 = abs2(x[1] - μ)
    for i = 2:length(x)
        @inbounds s2 += abs2(x[i] - μ)
    end
    ExgaussianKnownMuStats(g.μ, s2, length(x))
end

function suffstats(g::ExgaussianKnownMu, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    μ = g.μ
    s2 = abs2(x[1] - μ) * w[1]
    tw = w[1]
    for i = 2:length(x)
        @inbounds wi = w[i]
        @inbounds s2 += abs2(x[i] - μ) * wi
        tw += wi
    end
    ExgaussianKnownMuStats(g.μ, s2, tw)
end

struct ExgaussianKnownSigma <: IncompleteDistribution
    σ::Float64

    function ExgaussianKnownSigma(σ::Float64)
        σ > 0 || throw(ArgumentError("σ must be a positive value."))
        new(σ)
    end
end

struct ExgaussianKnownSigmaStats <: SufficientStats
    σ::Float64      # known std.dev
    sx::Float64      # (weighted) sum of x
    tw::Float64     # total sample weight
end

function suffstats(g::ExgaussianKnownSigma, x::AbstractArray{T}) where T<:Real
    ExgaussianKnownSigmaStats(g.σ, sum(x), Float64(length(x)))
end

function suffstats(g::ExgaussianKnownSigma, x::AbstractArray{T}, w::AbstractArray{T}) where T<:Real
    ExgaussianKnownSigmaStats(g.σ, dot(x, w), sum(w))
end

# fit_mle based on sufficient statistics

fit_mle(::Type{<:Exgaussian}, ss::ExgaussianStats) = Exgaussian(ss.m, sqrt(ss.s2 / ss.tw))
fit_mle(g::ExgaussianKnownMu, ss::ExgaussianKnownMuStats) = Exgaussian(g.μ, sqrt(ss.s2 / ss.tw))
fit_mle(g::ExgaussianKnownSigma, ss::ExgaussianKnownSigmaStats) = Exgaussian(ss.sx / ss.tw, g.σ)

# generic fit_mle methods

function fit_mle(::Type{<:Exgaussian}, x::AbstractArray{T}; mu::Float64=NaN, sigma::Float64=NaN) where T<:Real
    if isnan(mu)
        if isnan(sigma)
            fit_mle(Exgaussian, suffstats(Exgaussian, x))
        else
            g = ExgaussianKnownSigma(sigma)
            fit_mle(g, suffstats(g, x))
        end
    else
        if isnan(sigma)
            g = ExgaussianKnownMu(mu)
            fit_mle(g, suffstats(g, x))
        else
            Exgaussian(mu, sigma)
        end
    end
end

function fit_mle(::Type{<:Exgaussian}, x::AbstractArray{T}, w::AbstractArray{Float64}; mu::Float64=NaN, sigma::Float64=NaN) where T<:Real
    if isnan(mu)
        if isnan(sigma)
            fit_mle(Exgaussian, suffstats(Exgaussian, x, w))
        else
            g = ExgaussianKnownSigma(sigma)
            fit_mle(g, suffstats(g, x, w))
        end
    else
        if isnan(sigma)
            g = ExgaussianKnownMu(mu)
            fit_mle(g, suffstats(g, x, w))
        else
            Exgaussian(mu, sigma)
        end
    end
end

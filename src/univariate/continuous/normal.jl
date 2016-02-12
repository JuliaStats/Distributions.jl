immutable Normal{T<:Number} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    Normal(μ::T, σ::T) = new(μ, σ)
end

Normal{T<:Number}(μ::T, σ::T) = Normal{T}(μ, σ)
Normal(μ::Number, σ::Number) = Normal(promote(μ, σ)...)
Normal(μ::Number) = Normal(μ, one(μ))
#Normal(μ::Real, σ::Real) = (@check_args(Normal, σ > zero(σ)); Normal(μ, σ))
Normal(μ::Real) = Normal(μ, one(μ))
Normal() = Normal(0.0, 1.0)
Normal(μ::Complex, σ::Complex) = error("Normal with complex mean and variance not defined")

typealias Gaussian Normal

@distr_support Normal -Inf Inf


#### Parameters

params(d::Normal) = (d.μ, d.σ)


#### Statistics

mean(d::Normal) = d.μ
median(d::Normal) = d.μ
mode(d::Normal) = d.μ

var(d::Normal) = abs2(d.σ)
std(d::Normal) = d.σ
skewness(d::Normal) = 0.0
kurtosis(d::Normal) = 0.0

entropy(d::Normal) = 0.5 * (log2π + 1.0) + log(d.σ)


#### Evaluation

@_delegate_statsfuns Normal norm μ σ

gradlogpdf(d::Normal, x::Float64) = (d.μ - x) / d.σ^2

mgf(d::Normal, t::Real) = exp(t * d.μ + 0.5 * d.σ^2 * t^2)
cf(d::Normal, t::Real) = exp(im * t * d.μ - 0.5 * d.σ^2 * t^2)


#### Sampling

rand(d::Normal) = d.μ + d.σ * randn()


#### Fitting

immutable NormalStats <: SufficientStats
    s::Float64    # (weighted) sum of x
    m::Float64    # (weighted) mean of x
    s2::Float64   # (weighted) sum of (x - μ)^2
    tw::Float64    # total sample weight
end

function suffstats{T<:Real}(::Type{Normal}, x::AbstractArray{T})
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

function suffstats{T<:Real}(::Type{Normal}, x::AbstractArray{T}, w::AbstractArray{Float64})
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

immutable NormalKnownMu <: IncompleteDistribution
    μ::Float64
end

immutable NormalKnownMuStats <: SufficientStats
    μ::Float64      # known mean
    s2::Float64     # (weighted) sum of (x - μ)^2
    tw::Float64     # total sample weight
end

function suffstats{T<:Real}(g::NormalKnownMu, x::AbstractArray{T})
    μ = g.μ
    s2 = abs2(x[1] - μ)
    for i = 2:length(x)
        @inbounds s2 += abs2(x[i] - μ)
    end
    NormalKnownMuStats(g.μ, s2, length(x))
end

function suffstats{T<:Real}(g::NormalKnownMu, x::AbstractArray{T}, w::AbstractArray{Float64})
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


immutable NormalKnownSigma <: IncompleteDistribution
    σ::Float64

    function NormalKnownSigma(σ::Float64)
        σ > 0.0 || throw(ArgumentError("σ must be a positive value."))
        new(σ)
    end
end

immutable NormalKnownSigmaStats <: SufficientStats
    σ::Float64      # known std.dev
    sx::Float64      # (weighted) sum of x
    tw::Float64     # total sample weight
end

function suffstats{T<:Real}(g::NormalKnownSigma, x::AbstractArray{T})
    NormalKnownSigmaStats(g.σ, sum(x), Float64(length(x)))
end

function suffstats{T<:Real}(g::NormalKnownSigma, x::AbstractArray{T}, w::AbstractArray{T})
    NormalKnownSigmaStats(g.σ, dot(x, w), sum(w))
end

# fit_mle based on sufficient statistics

fit_mle(::Type{Normal}, ss::NormalStats) = Normal(ss.m, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownMu, ss::NormalKnownMuStats) = Normal(g.μ, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownSigma, ss::NormalKnownSigmaStats) = Normal(ss.sx / ss.tw, g.σ)

# generic fit_mle methods

function fit_mle{T<:Real}(::Type{Normal}, x::AbstractArray{T}; mu::Float64=NaN, sigma::Float64=NaN)
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

function fit_mle{T<:Real}(::Type{Normal}, x::AbstractArray{T}, w::AbstractArray{Float64}; mu::Float64=NaN, sigma::Float64=NaN)
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

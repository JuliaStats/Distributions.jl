immutable Normal <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64
    function Normal(μ::Real, σ::Real)
    	σ > zero(σ) || error("std.dev. must be positive")
    	new(float64(μ), float64(σ))
    end
end
Normal(μ::Real) = Normal(float64(μ), 1.0)
Normal() = Normal(0.0, 1.0)

@_jl_dist_2p Normal norm

@continuous_distr_support Normal -Inf Inf

const Gaussian = Normal

zval(d::Normal, x::Real) = (x - d.μ)/d.σ
xval(d::Normal, z::Real) = d.μ + d.σ * z

pdf(d::Normal, x::Real) = φ(zval(d,x))/d.σ
logpdf(d::Normal, x::Real) = logφ(zval(d,x)) - log(d.σ)

cdf(d::Normal, x::Real) = Φ(zval(d,x))
ccdf(d::Normal, x::Real) = Φc(zval(d,x))
logcdf(d::Normal, x::Real) = logΦ(zval(d,x))
logccdf(d::Normal, x::Real) = logΦc(zval(d,x))    

quantile(d::Normal, p::Real) = xval(d, Φinv(p))
cquantile(d::Normal, p::Real) = xval(d, -Φinv(p))
invlogcdf(d::Normal, p::Real) = xval(d, logΦinv(p))
invlogccdf(d::Normal, p::Real) = xval(d, -logΦinv(p))

entropy(d::Normal) = 0.5 * (log2π + 1.) + log(d.σ)

kurtosis(d::Normal) = 0.0

mean(d::Normal) = d.μ

median(d::Normal) = d.μ

mgf(d::Normal, t::Real) = exp(t * d.μ + 0.5 * d.σ^2 * t^2)

cf(d::Normal, t::Real) = exp(im * t * d.μ - 0.5 * d.σ^2 * t^2)

mode(d::Normal) = d.μ
modes(d::Normal) = [d.μ]

rand(d::Normal) = d.μ + d.σ * randn()

skewness(d::Normal) = 0.0

std(d::Normal) = d.σ

var(d::Normal) = d.σ^2

## Fit model

immutable NormalStats <: SufficientStats
    s::Float64    # (weighted) sum of x
    m::Float64    # (weighted) mean of x
    s2::Float64   # (weighted) sum of (x - μ)^2
    tw::Float64    # total sample weight

    NormalStats(s::Real, m::Real, s2::Real, tw::Real) = new(float64(s), float64(m), float64(s2), float64(tw))
end

function suffstats{T<:Real}(::Type{Normal}, x::Array{T}) 
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

function suffstats{T<:Real}(::Type{Normal}, x::Array{T}, w::Array{Float64}) 
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

function suffstats{T<:Real}(g::NormalKnownMu, x::Array{T})
    μ = g.μ
    s2 = abs2(x[1] - μ)
    for i = 2:length(x)
        @inbounds s2 += abs2(x[i] - μ)
    end
    NormalKnownMuStats(g.μ, s2, float64(length(x)))
end

function suffstats{T<:Real}(g::NormalKnownMu, x::Array{T}, w::Array{Float64})
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
    s::Float64      # (weighted) sum of x
    tw::Float64     # total sample weight
end

function suffstats{T<:Real}(g::NormalKnownSigma, x::Array{T})
    NormalKnownSigmaStats(g.σ, sum(x), float64(length(x)))    
end

function suffstats{T<:Real}(g::NormalKnownSigma, x::Array{T}, w::Array{T})
    NormalKnownSigmaStats(g.σ, dot(x, w), sum(w))    
end

# fit_mle based on sufficient statistics

fit_mle(::Type{Normal}, ss::NormalStats) = Normal(ss.m, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownMu, ss::NormalKnownMuStats) = Normal(g.μ, ss.s2 / ss.tw)
fit_mle(g::NormalKnownSigma, ss::NormalKnownSigmaStats) = Normal(ss.s / ss.tw, g.σ)

# generic fit_mle methods

function fit_mle{T<:Real}(::Type{Normal}, x::Array{T}; mu::Float64=NaN, sigma::Float64=NaN)
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

function fit_mle{T<:Real}(::Type{Normal}, x::Array{T}, w::Array{Float64}; mu::Float64=NaN, sigma::Float64=NaN)
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


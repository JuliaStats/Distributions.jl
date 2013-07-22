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

const Gaussian = Normal

begin
    zval(d::Normal, x::Real) = (x - d.μ)/d.σ
    xval(d::Normal, z::Real) = d.μ + d.σ * z
    
    φ{T<:FloatingPoint}(z::T) = exp(-0.5*z*z)/√2π
    pdf(d::Normal, x::FloatingPoint) = φ(zval(d,x))/d.σ

    logφ{T<:FloatingPoint}(z::T) = -0.5*(z*z + log2π)
    logpdf(d::Normal, x::FloatingPoint) = logφ(zval(d,x)) - log(d.σ)

    Φ{T<:FloatingPoint}(z::T) = 0.5 + 0.5*erf(z/√2)
    cdf(d::Normal, x::FloatingPoint) = Φ(zval(d,x))

    Φc{T<:FloatingPoint}(z::T) = 0.5*erfc(z/√2)
    ccdf(d::Normal, x::FloatingPoint) = Φc(zval(d,x))

    Φinv{T<:FloatingPoint}(p::T) = √2 * erfinv(2p - 1)
    quantile(d::Normal, p::FloatingPoint) = xval(d, Φinv(p))

    Φcinv{T<:FloatingPoint}(p::T) = √2 * erfcinv(2p)
    cquantile(d::Normal, p::FloatingPoint) = xval(d, Φcinv(p))
end

for f in [pdf, logpdf, ccdf, cdf, quantile]
    quote
        $f(d::Normal, x::Integer) = $f(d, float64(x))
    end
end

entropy(d::Normal) = 0.5 * (log2π + 1.) + log(d.σ)

insupport(d::Normal, x::Real) = isfinite(x)

kurtosis(d::Normal) = 0.0

mean(d::Normal) = d.μ

median(d::Normal) = d.μ

mgf(d::Normal, t::Real) = exp(t * d.μ + 0.5 * d.σ^t * t^2)

cf(d::Normal, t::Real) = exp(im * t * d.μ - 0.5 * d.σ^t * t^2)

modes(d::Normal) = [d.μ]

rand(d::Normal) = d.μ + d.σ * randn()

skewness(d::Normal) = 0.0

std(d::Normal) = d.σ

var(d::Normal) = d.σ^2

## Fit model

immutable NormalStats
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

    # compute ss
    s2 = abs2(x[1] - m)
    for i = 2:n
        s2 += abs2(x[i] - m)  # is there a reason this is not also @inbounds ?
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

    # compute ss
    s2 = w[1] * abs2(x[1] - m)
    for i = 2:n
        @inbounds s2 += abs2(x[i] - m)
    end

    NormalStats(s, m, s2, tw)
end

fit_mle(::Type{Normal}, ss::NormalStats) = Normal(ss.m, sqrt(ss.s2 / ss.tw))


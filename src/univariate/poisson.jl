immutable Poisson <: DiscreteUnivariateDistribution
    lambda::Float64
    function Poisson(l::Real)
    	l > zero(l) || error("lambda must be positive")
        new(float64(l))
    end
    Poisson() = new(1.0)
end

@_jl_dist_1p Poisson pois

## Support
isupperbounded(::Union(Poisson, Type{Poisson})) = false
islowerbounded(::Union(Poisson, Type{Poisson})) = true
isbounded(::Union(Poisson, Type{Poisson})) = false

minimum(::Union(Poisson, Type{Poisson})) = 0
maximum(::Union(Poisson, Type{Poisson})) = Inf

insupport(::Poisson, x::Real) = isinteger(x) && zero(x) <= x
insupport(::Type{Poisson}, x::Real) = isinteger(x) && zero(x) <= x

## Properties
mean(d::Poisson) = d.lambda
median(d::Poisson) = quantile(d, 0.5)
mode(d::Poisson) = ifloor(d.lambda)
modes(d::Poisson) = [mode(d)]

var(d::Poisson) = d.lambda
skewness(d::Poisson) = 1.0 / sqrt(d.lambda)
kurtosis(d::Poisson) = 1.0 / d.lambda

function entropy(d::Poisson)
    λ = d.lambda
    if λ < 50.0
        s = 0.0
        for k in 1:100
            s += λ^k * lgamma(k + 1.0) / gamma(k + 1.0)
        end
        return λ * (1.0 - log(λ)) + exp(-λ) * s
    else
        return 0.5 * log(2 * pi * e * λ) -
               (1 / (12 * λ)) -
               (1 / (24 * λ * λ)) -
               (19 / (360 * λ * λ * λ))
    end
end

## Functions
function pdf(d::Poisson, x::Real)
    λ = d.lambda
    if !(isinteger(x) && x >= zero(x))
        0.0
    elseif x == zero(x)
        exp(-λ)
    elseif x < 10
        exp(x*log(λ) - λ) / gamma(x+1.0)
    else
        exp(x*logmxp1(λ/x)-lstirling_asym(x)) / sqrt(twoπ*x)
    end
end
function logpdf(d::Poisson, x::Real)
    λ = d.lambda
    if !(isinteger(x) && x >= zero(x))
        -Inf
    elseif x == zero(x)
        -λ
    elseif x < 10
        x*log(λ) - λ - lgamma(x+1.0)
    else
        x*logmxp1(λ/x) - lstirling_asym(x) - 0.5*log(x) - 0.5*log2π
    end
end

mgf(d::Poisson, t::Real) = exp(d.lambda*expm1(t))
cf(d::Poisson, t::Real) = exp(d.lambda*(exp(im*t)-1.0))

## Fitting
immutable PoissonStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    tw::Float64   # total sample weight
end

suffstats(::Type{Poisson}, x::Array) = PoissonStats(float64(sum(x)), float64(length(x)))

function suffstats(::Type{Poisson}, x::Array, w::Array{Float64})
    n = length(x)
    n == length(w) || throw(ArgumentError("Inconsistent array lengths."))
    sx = 0.
    tw = 0.
    for i = 1 : n
        @inbounds wi = w[i]
        @inbounds sx += x[i] * wi
        tw += wi
    end
    PoissonStats(sx, tw)
end

fit_mle(::Type{Poisson}, ss::PoissonStats) = Poisson(ss.sx / ss.tw)


immutable Poisson <: DiscreteUnivariateDistribution
    lambda::Float64
    function Poisson(l::Real)
    	l > zero(l) || error("lambda must be positive")
        new(float64(l))
    end
    Poisson() = new(1.0)
end

@_jl_dist_1p Poisson pois

@distr_support Poisson 0 Inf

mean(d::Poisson) = d.lambda

median(d::Poisson) = quantile(d, 0.5)

mode(d::Poisson) = ifloor(d.lambda)
modes(d::Poisson) = isinteger(d.lambda) ? [int(d.lambda)-1,int(d.lambda)] : [ifloor(d.lambda)]

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

immutable RecursivePoissonProbEvaluator <: RecursiveProbabilityEvaluator
    λ::Float64
end

RecursivePoissonProbEvaluator(d::Poisson) = RecursivePoissonProbEvaluator(d.lambda)
nextpdf(s::RecursivePoissonProbEvaluator, p::Float64, x::Integer) = p * s.λ / x
_pdf!(r::AbstractArray, d::Poisson, rgn::UnitRange) = _pdf!(r, d, rgn, RecursivePoissonProbEvaluator(d))


function mgf(d::Poisson, t::Real)
    l = d.lambda
    return exp(l * (exp(t) - 1.0))
end

function cf(d::Poisson, t::Real)
    l = d.lambda
    return exp(l * (exp(im * t) - 1.0))
end

# model fitting

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


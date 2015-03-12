immutable Poisson <: DiscreteUnivariateDistribution
    λ::Float64

    function Poisson(λ::Float64)
        λ > 0.0 || error("λ must be positive.")
        new(λ)
    end

    @compat Poisson(λ::Real) = Poisson(Float64(λ))
    Poisson() = new(1.0)
end

@_jl_dist_1p Poisson pois

@distr_support Poisson 0 Inf


### Parameters

params(d::Poisson) = (d.λ,)

rate(d::Poisson) = d.λ


### Statistics

mean(d::Poisson) = d.λ

mode(d::Poisson) = floor(Int,d.λ)

function modes(d::Poisson)
    λ = d.λ
    @compat isinteger(λ) ? [round(Int, λ)-1, round(Int, λ)] : [floor(Int, λ)]
end

var(d::Poisson) = d.λ

skewness(d::Poisson) = 1.0 / sqrt(d.λ)

kurtosis(d::Poisson) = 1.0 / d.λ

function entropy(d::Poisson)
    λ = rate(d)
    if λ < 50.0
        s = 0.0
        λk = 1.0
        for k = 1:100
            λk *= λ
            s += λk * lgamma(k + 1.0) / gamma(k + 1.0)
        end
        return λ * (1.0 - log(λ)) + exp(-λ) * s
    else
        return 0.5 * log(2 * pi * e * λ) -
               (1 / (12 * λ)) -
               (1 / (24 * λ * λ)) -
               (19 / (360 * λ * λ * λ))
    end
end


### Evaluation

immutable RecursivePoissonProbEvaluator <: RecursiveProbabilityEvaluator
    λ::Float64
end

RecursivePoissonProbEvaluator(d::Poisson) = RecursivePoissonProbEvaluator(rate(d))
nextpdf(s::RecursivePoissonProbEvaluator, p::Float64, x::Integer) = p * s.λ / x
_pdf!(r::AbstractArray, d::Poisson, rgn::UnitRange) = _pdf!(r, d, rgn, RecursivePoissonProbEvaluator(d))

function mgf(d::Poisson, t::Real)
    λ = rate(d)
    return exp(λ * (exp(t) - 1.0))
end

function cf(d::Poisson, t::Real)
    λ = rate(d)
    return exp(λ * (cis(t) - 1.0))
end


### Fitting

immutable PoissonStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    tw::Float64   # total sample weight
end

@compat suffstats(::Type{Poisson}, x::Array) = PoissonStats(Float64(sum(x)), Float64(length(x)))

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


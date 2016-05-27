doc"""
    Poisson(λ)

A *Poisson distribution* descibes the number of independent events occurring within a unit time interval, given the average rate of occurrence `λ`.

$P(X = k) = \frac{\lambda^k}{k!} e^{-\lambda}, \quad \text{ for } k = 0,1,2,\ldots.$

```julia
Poisson()        # Poisson distribution with rate parameter 1
Poisson(lambda)       # Poisson distribution with rate parameter lambda

params(d)        # Get the parameters, i.e. (λ,)
mean(d)          # Get the mean arrival rate, i.e. λ
```

External links:

* [Poisson distribution on Wikipedia](http://en.wikipedia.org/wiki/Poisson_distribution)

"""
immutable Poisson{T<:Real} <: DiscreteUnivariateDistribution
    λ::T

    Poisson(λ::Real) = (@check_args(Poisson, λ >= zero(λ)); new(λ))
end

Poisson{T<:Real}(λ::T) = Poisson{T}(λ)
Poisson() = Poisson(1.0)

@distr_support Poisson 0 (d.λ == zero(typeof(d.λ)) ? 0 : Inf)

# #### Conversions
convert{T <: Real}(::Type{Poisson{T}}, λ::Real) = Poisson(T(λ))
convert{T <: Real, S <: Real}(::Type{Poisson{T}}, d::Poisson{S}) = Poisson(T(d.λ))

### Parameters

params(d::Poisson) = (d.λ,)

rate(d::Poisson) = d.λ


### Statistics

mean(d::Poisson) = d.λ

mode(d::Poisson) = floor(Int,d.λ)

function modes(d::Poisson)
    λ = d.λ
    isinteger(λ) ? [round(Int, λ)-1, round(Int, λ)] : [floor(Int, λ)]
end

var(d::Poisson) = d.λ

skewness(d::Poisson) = one(typeof(d.λ)) / sqrt(d.λ)

kurtosis(d::Poisson) = one(typeof(d.λ)) / d.λ

function entropy(d::Poisson)
    λ = rate(d)
    if λ == zero(typeof(λ))
        return 0.0
    elseif λ < 50.0
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

@_delegate_statsfuns Poisson pois λ

rand(d::Poisson) = convert(Int, StatsFuns.Rmath.poisrand(d.λ))

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

suffstats{T<:Integer}(::Type{Poisson}, x::AbstractArray{T}) = PoissonStats(sum(x), length(x))

function suffstats{T<:Integer}(::Type{Poisson}, x::AbstractArray{T}, w::AbstractArray{Float64})
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

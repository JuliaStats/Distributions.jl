"""
    Poisson(λ)

A *Poisson distribution* describes the number of independent events occurring within a unit time interval, given the average rate of occurrence `λ`.

```math
P(X = k) = \\frac{\\lambda^k}{k!} e^{-\\lambda}, \\quad \\text{ for } k = 0,1,2,\\ldots.
```

```julia
Poisson()        # Poisson distribution with rate parameter 1
Poisson(lambda)       # Poisson distribution with rate parameter lambda

params(d)        # Get the parameters, i.e. (λ,)
mean(d)          # Get the mean arrival rate, i.e. λ
```

External links:

* [Poisson distribution on Wikipedia](http://en.wikipedia.org/wiki/Poisson_distribution)

"""
struct Poisson{T<:Real} <: DiscreteUnivariateDistribution
    λ::T

    Poisson{T}(λ::Real) where {T <: Real} = new{T}(λ)
end

function Poisson(λ::Real; check_args::Bool=true)
    @check_args Poisson (λ, λ >= zero(λ))
    return Poisson{typeof(λ)}(λ)
end

Poisson(λ::Integer; check_args::Bool=true) = Poisson(float(λ); check_args=check_args)
Poisson() = Poisson{Float64}(1.0)

@distr_support Poisson 0 (d.λ == zero(typeof(d.λ)) ? 0 : Inf)

#### Conversions
convert(::Type{Poisson{T}}, λ::S) where {T <: Real, S <: Real} = Poisson(T(λ))
Base.convert(::Type{Poisson{T}}, d::Poisson) where {T<:Real} = Poisson{T}(T(d.λ))
Base.convert(::Type{Poisson{T}}, d::Poisson{T}) where {T<:Real} = d

### Parameters

params(d::Poisson) = (d.λ,)
partype(::Poisson{T}) where {T} = T

rate(d::Poisson) = d.λ

### Statistics

mean(d::Poisson) = d.λ

mode(d::Poisson) = floor(Int,d.λ)

function modes(d::Poisson)
    λ = d.λ
    isinteger(λ) ? [round(Int, λ) - 1, round(Int, λ)] : [floor(Int, λ)]
end

var(d::Poisson) = d.λ

skewness(d::Poisson) = one(typeof(d.λ)) / sqrt(d.λ)

kurtosis(d::Poisson) = one(typeof(d.λ)) / d.λ

function entropy(d::Poisson{T}) where T<:Real
    λ = rate(d)
    if λ == zero(T)
        return zero(T)
    elseif λ < 50
        s = zero(T)
        λk = one(T)
        for k = 1:100
            λk *= λ
            s += λk * loggamma(k + 1) / gamma(k + 1)
        end
        return λ * (1 - log(λ)) + exp(-λ) * s
    else
        return log(2 * pi * ℯ * λ)/2 -
               (1 / (12 * λ)) -
               (1 / (24 * λ * λ)) -
               (19 / (360 * λ * λ * λ))
    end
end

function kldivergence(p::Poisson, q::Poisson)
    λp = rate(p)
    λq = rate(q)
    # `false` is a strong zero and ensures that `λp = 0` is handled correctly
    # we don't use `xlogy` since it returns `NaN` for `λp = λq = 0`
    return λq - λp + (λp > 0) * (λp * log(λp / λq))
end

### Evaluation

@_delegate_statsfuns Poisson pois λ

function mgf(d::Poisson, t::Real)
    λ = rate(d)
    return exp(λ * (exp(t) - 1))
end
cgf(d::Poisson, t) = mean(d) * expm1(t)

function cf(d::Poisson, t::Real)
    λ = rate(d)
    return exp(λ * (cis(t) - 1))
end


### Fitting

struct PoissonStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    tw::Float64   # total sample weight
end

suffstats(::Type{<:Poisson}, x::AbstractArray{T}) where {T<:Integer} = PoissonStats(sum(x), length(x))

function suffstats(::Type{<:Poisson}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    n = length(x)
    n == length(w) || throw(DimensionMismatch("Inconsistent array lengths."))
    sx = 0.
    tw = 0.
    for i in eachindex(x, w)
        @inbounds wi = w[i]
        @inbounds sx += x[i] * wi
        tw += wi
    end
    PoissonStats(sx, tw)
end

fit_mle(::Type{<:Poisson}, ss::PoissonStats) = Poisson(ss.sx / ss.tw)

## samplers

const poissonsampler_threshold = 6

function sampler(d::Poisson)
    if rate(d) < poissonsampler_threshold
        return PoissonCountSampler(d)
    else
        return PoissonADSampler(d)
    end
end

rand(rng::AbstractRNG, d::Poisson) = rand(rng, sampler(d))

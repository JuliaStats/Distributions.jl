"""
    Gamma(α,θ)

The *Gamma distribution* with shape parameter `α` and scale `θ` has probability density
function

```math
f(x; \\alpha, \\theta) = \\frac{x^{\\alpha-1} e^{-x/\\theta}}{\\Gamma(\\alpha) \\theta^\\alpha},
\\quad x > 0
```

```julia
Gamma()          # Gamma distribution with unit shape and unit scale, i.e. Gamma(1, 1)
Gamma(α)         # Gamma distribution with shape α and unit scale, i.e. Gamma(α, 1)
Gamma(α, θ)      # Gamma distribution with shape α and scale θ

params(d)        # Get the parameters, i.e. (α, θ)
shape(d)         # Get the shape parameter, i.e. α
scale(d)         # Get the scale parameter, i.e. θ
```

External links

* [Gamma distribution on Wikipedia](http://en.wikipedia.org/wiki/Gamma_distribution)

"""
struct Gamma{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    θ::T
    Gamma{T}(α, θ) where {T} = new{T}(α, θ)
end

function Gamma(α::T, θ::T; check_args::Bool=true) where {T <: Real}
    @check_args Gamma (α, α > zero(α)) (θ, θ > zero(θ))
    return Gamma{T}(α, θ)
end

Gamma(α::Real, θ::Real; check_args::Bool=true) = Gamma(promote(α, θ)...; check_args=check_args)
Gamma(α::Integer, θ::Integer; check_args::Bool=true) = Gamma(float(α), float(θ); check_args=check_args)
Gamma(α::Real; check_args::Bool=true) = Gamma(α, one(α); check_args=check_args)
Gamma() = Gamma{Float64}(1.0, 1.0)

@distr_support Gamma 0.0 Inf

#### Conversions
convert(::Type{Gamma{T}}, α::S, θ::S) where {T <: Real, S <: Real} = Gamma(T(α), T(θ))
Base.convert(::Type{Gamma{T}}, d::Gamma) where {T<:Real} = Gamma{T}(T(d.α), T(d.θ))
Base.convert(::Type{Gamma{T}}, d::Gamma{T}) where {T<:Real} = d

#### Parameters

shape(d::Gamma) = d.α
scale(d::Gamma) = d.θ
rate(d::Gamma) = 1 / d.θ

params(d::Gamma) = (d.α, d.θ)
partype(::Gamma{T}) where {T} = T

#### Statistics

mean(d::Gamma) = d.α * d.θ

var(d::Gamma) = d.α * d.θ^2

skewness(d::Gamma) = 2 / sqrt(d.α)

kurtosis(d::Gamma) = 6 / d.α

function mode(d::Gamma)
    (α, θ) = params(d)
    α >= 1 ? θ * (α - 1) : error("Gamma has no mode when shape < 1")
end

function entropy(d::Gamma)
    (α, θ) = params(d)
    α + loggamma(α) + (1 - α) * digamma(α) + log(θ)
end

mgf(d::Gamma, t::Real) = (1 - t * d.θ)^(-d.α)
function cgf(d::Gamma, t)
    α, θ = params(d)
    return α * cgf(Exponential{typeof(θ)}(θ), t)
end

cf(d::Gamma, t::Real) = (1 - im * t * d.θ)^(-d.α)

function kldivergence(p::Gamma, q::Gamma)
    # We use the parametrization with the scale θ
    αp, θp = params(p)
    αq, θq = params(q)
    θp_over_θq = θp / θq
    return (αp - αq) * digamma(αp) - loggamma(αp) + loggamma(αq) -
        αq * log(θp_over_θq) + αp * (θp_over_θq - 1)
end

#### Evaluation & Sampling

@_delegate_statsfuns Gamma gamma α θ

gradlogpdf(d::Gamma{T}, x::Real) where {T<:Real} =
    insupport(Gamma, x) ? (d.α - 1) / x - 1 / d.θ : zero(T)

function rand(rng::AbstractRNG, d::Gamma)
    if shape(d) < 1.0
        # TODO: shape(d) = 0.5 : use scaled chisq
        return rand(rng, GammaIPSampler(d))
    elseif shape(d) == 1.0
        θ = 
        return rand(rng, Exponential{partype(d)}(scale(d)))
    else
        return rand(rng, GammaMTSampler(d))
    end
end

function sampler(d::Gamma)
    if shape(d) < 1.0
        # TODO: shape(d) = 0.5 : use scaled chisq
        return GammaIPSampler(d)
    elseif shape(d) == 1.0
        return sampler(Exponential{partype(d)}(scale(d)))
    else
        return GammaMTSampler(d)
    end
end

#### Fit model

struct GammaStats <: SufficientStats
    sx::Float64      # (weighted) sum of x
    slogx::Float64   # (weighted) sum of log(x)
    tw::Float64      # total sample weight

    GammaStats(sx::Real, slogx::Real, tw::Real) = new(sx, slogx, tw)
end

function suffstats(::Type{<:Gamma}, x::AbstractArray{T}) where T<:Real
    sx = zero(T)
    slogx = zero(T)
    for xi = x
        sx += xi
        slogx += log(xi)
    end
    GammaStats(sx, slogx, length(x))
end

function suffstats(::Type{<:Gamma}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    n = length(x)
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    end

    sx = zero(T)
    slogx = zero(T)
    tw = zero(T)
    for i in eachindex(x, w)
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        sx += wi * xi
        slogx += wi * log(xi)
        tw += wi
    end
    GammaStats(sx, slogx, tw)
end

function gamma_mle_update(logmx::Float64, mlogx::Float64, a::Float64)
    ia = 1 / a
    z = ia + (mlogx - logmx + log(a) - digamma(a)) / (abs2(a) * (ia - trigamma(a)))
    1 / z
end

function fit_mle(::Type{<:Gamma}, ss::GammaStats;
    alpha0::Float64=NaN, maxiter::Int=1000, tol::Float64=1e-16)

    mx = ss.sx / ss.tw
    logmx = log(mx)
    mlogx = ss.slogx / ss.tw

    a::Float64 = isnan(alpha0) ? (logmx - mlogx)/2 : alpha0
    converged = false

    t = 0
    while !converged && t < maxiter
        t += 1
        a_old = a
        a = gamma_mle_update(logmx, mlogx, a)
        converged = abs(a - a_old) <= tol
    end

    Gamma(a, mx / a)
end

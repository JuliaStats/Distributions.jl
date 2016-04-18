doc"""
    Gamma(α,θ)

The *Gamma distribution* with shape parameter `α` and scale `θ` has probability density
function

$f(x; \alpha, \beta) = \frac{x^{\alpha-1} e^{-x/\beta}}{\Gamma(\alpha) \beta^\alpha},
\quad x > 0$

```julia
Gamma()          # Gamma distribution with unit shape and unit scale, i.e. Gamma(1.0, 1.0)
Gamma(a)         # Gamma distribution with shape a and unit scale, i.e. Gamma(a, 1.0)
Gamma(a, b)      # Gamma distribution with shape a and scale b

params(d)        # Get the parameters, i.e. (a, b)
shape(d)         # Get the shape parameter, i.e. a
scale(d)         # Get the scale parameter, i.e. b
```

External links

* [Gamma distribution on Wikipedia](http://en.wikipedia.org/wiki/Gamma_distribution)

"""
immutable Gamma{T <: Real} <: ContinuousUnivariateDistribution
    α::T
    θ::T

    function Gamma(α, θ)
        @check_args(Gamma, α > zero(α) && θ > zero(θ))
        new(α, θ)
    end
end

Gamma{T<:Real}(α::T, θ::T) = Gamma{T}(α, θ)
Gamma(α::Real, θ::Real) = Gamma(promote(α, θ)...)
Gamma(α::Real) = Gamma(α, 1.0)
Gamma() = Gamma(1.0, 1.0)

@distr_support Gamma 0.0 Inf

#### Conversions
convert{T <: Real, S <: Real}(::Type{Gamma{T}}, α::S, θ::S) = Gamma(T(α), T(θ))
convert{T <: Real, S <: Real}(::Type{Gamma{T}}, d::Gamma{S}) = Gamma(T(d.α), T(d.θ))

#### Parameters

shape(d::Gamma) = d.α
scale(d::Gamma) = d.θ
rate(d::Gamma) = 1.0 / d.θ

params(d::Gamma) = (d.α, d.θ)


#### Statistics

mean(d::Gamma) = d.α * d.θ

var(d::Gamma) = d.α * d.θ^2

skewness(d::Gamma) = 2.0 / sqrt(d.α)

kurtosis(d::Gamma) = 6.0 / d.α

function mode(d::Gamma)
    (α, θ) = params(d)
    α >= 1.0 ? θ * (α - 1.0) : error("Gamma has no mode when shape < 1.0")
end

function entropy(d::Gamma)
    (α, θ) = params(d)
    α + lgamma(α) + (1.0 - α) * digamma(α) + log(θ)
end

mgf(d::Gamma, t::Real) = (1.0 - t * d.θ)^(-d.α)

cf(d::Gamma, t::Real) = (1.0 - im * t * d.θ)^(-d.α)


#### Evaluation & Sampling

@_delegate_statsfuns Gamma gamma α θ

gradlogpdf(d::Gamma, x::Float64) =
    insupport(Gamma, x) ? (d.α - 1.0) / x - 1.0 / d.θ : 0.0

rand(d::Gamma) = StatsFuns.Rmath.gammarand(d.α, d.θ)


#### Fit model

immutable GammaStats <: SufficientStats
    sx::Float64      # (weighted) sum of x
    slogx::Float64   # (weighted) sum of log(x)
    tw::Float64      # total sample weight

    GammaStats(sx::Real, slogx::Real, tw::Real) = new(sx, slogx, tw)
end

function suffstats{T<:Real}(::Type{Gamma}, x::AbstractArray{T})
    sx = 0.
    slogx = 0.
    for xi = x
        sx += xi
        slogx += log(xi)
    end
    GammaStats(sx, slogx, length(x))
end

function suffstats{T<:Real}(::Type{Gamma}, x::AbstractArray{T}, w::AbstractArray{Float64})
    n = length(x)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    sx = 0.
    slogx = 0.
    tw = 0.
    for i = 1:n
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        sx += wi * xi
        slogx += wi * log(xi)
        tw += wi
    end
    GammaStats(sx, slogx, tw)
end

function gamma_mle_update(logmx::Float64, mlogx::Float64, a::Float64)
    ia = 1.0 / a
    z = ia + (mlogx - logmx + log(a) - digamma(a)) / (abs2(a) * (ia - trigamma(a)))
    1.0 / z
end

function fit_mle(::Type{Gamma}, ss::GammaStats;
    alpha0::Float64=NaN, maxiter::Int=1000, tol::Float64=1.0e-16)

    mx = ss.sx / ss.tw
    logmx = log(mx)
    mlogx = ss.slogx / ss.tw

    a::Float64 = isnan(alpha0) ? 0.5 / (logmx - mlogx) : alpha0
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

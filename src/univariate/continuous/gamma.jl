"""
    Gamma <: ContinuousUnivariateDistribution

The *gamma* probability distribution.

# Constructors

    Gamma(α|alpha|shape=1, θ|theta|scale=1)

Construct a `Gamma` distribution object with shape `α` and scale `θ`.

    Gamma(α|alpha|shape=1, β|beta|rate=1)

Construct a `Gamma` distribution object with shape `α` and rate `β = 1/θ`.

    Gamma(mean=, α|alpha|shape=)
    Gamma(mean=, θ|theta|scale=)
    Gamma(mean=, β|beta|rate=)
    Gamma(mean=, std=)
    Gamma(mean=, var=)

Construct a `Gamma` distribution object matching the relevant moments and parameters.

# Details
A gamma distribution with shape parameter `α` and scale `θ` has probability density
function

```math
f(x; \\alpha, \\theta) = \\frac{x^{\\alpha-1} e^{-x/\\theta}}{\\Gamma(\\alpha) \\theta^\\alpha},
\\quad x > 0
```

# Examples
```julia
Gamma()
Gamma(α=3)
Gamma(α=3, θ=2)
```

# External links

* [Gamma distribution on Wikipedia](http://en.wikipedia.org/wiki/Gamma_distribution)

"""
struct Gamma{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    θ::T

    function Gamma{T}(α, θ) where T
        @check_args(Gamma, α > zero(α) && θ > zero(θ))
        new{T}(α, θ)
    end
end

Gamma(α::T, θ::T) where {T<:Real} = Gamma{T}(α, θ)
Gamma(α::Real, θ::Real) = Gamma(promote(α, θ)...)
Gamma(α::Integer, θ::Integer) = Gamma(float(α), float(θ))

@kwdispatch (::Type{D})(;alpha=>α, shape=>α, theta=>θ, scale=>θ, beta=>β, rate=>β) where {D<:Gamma} begin
    () -> D(1,1)
    (α) -> D(α,1)
    
    (θ) -> D(1,θ)
    (α,θ) -> D(α,θ)

    (β) -> D(1,inv(β))
    (α,β) -> D(α,inv(β))
    
    (mean, α) -> D(α, mean/α)
    (mean, θ) -> D(mean/θ, θ)

    function (mean, std)
        θ=std^2/mean
        D(mean/θ, θ)
    end
    function (mean, var)
        θ=var/mean
        D(mean/θ, θ)
    end
end

@distr_support Gamma 0.0 Inf

#### Conversions
convert(::Type{Gamma{T}}, α::S, θ::S) where {T <: Real, S <: Real} = Gamma(T(α), T(θ))
convert(::Type{Gamma{T}}, d::Gamma{S}) where {T <: Real, S <: Real} = Gamma(T(d.α), T(d.θ))

#### Parameters

shape(d::Gamma) = d.α
scale(d::Gamma) = d.θ
rate(d::Gamma) = 1 / d.θ

params(d::Gamma) = (d.α, d.θ)
@inline partype(d::Gamma{T}) where {T<:Real} = T


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
    α + lgamma(α) + (1 - α) * digamma(α) + log(θ)
end

mgf(d::Gamma, t::Real) = (1 - t * d.θ)^(-d.α)

cf(d::Gamma, t::Real) = (1 - im * t * d.θ)^(-d.α)


#### Evaluation & Sampling

@_delegate_statsfuns Gamma gamma α θ

gradlogpdf(d::Gamma{T}, x::Real) where {T<:Real} =
    insupport(Gamma, x) ? (d.α - 1) / x - 1 / d.θ : zero(T)

rand(d::Gamma) = StatsFuns.RFunctions.gammarand(d.α, d.θ)


#### Fit model

struct GammaStats <: SufficientStats
    sx::Float64      # (weighted) sum of x
    slogx::Float64   # (weighted) sum of log(x)
    tw::Float64      # total sample weight

    GammaStats(sx::Real, slogx::Real, tw::Real) = new(sx, slogx, tw)
end

function suffstats(::Type{Gamma}, x::AbstractArray{T}) where T<:Real
    sx = zero(T)
    slogx = zero(T)
    for xi = x
        sx += xi
        slogx += log(xi)
    end
    GammaStats(sx, slogx, length(x))
end

function suffstats(::Type{Gamma}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    n = length(x)
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    end

    sx = zero(T)
    slogx = zero(T)
    tw = zero(T)
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
    ia = 1 / a
    z = ia + (mlogx - logmx + log(a) - digamma(a)) / (abs2(a) * (ia - trigamma(a)))
    1 / z
end

function fit_mle(::Type{Gamma}, ss::GammaStats;
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

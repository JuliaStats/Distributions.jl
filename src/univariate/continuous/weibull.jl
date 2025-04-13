"""
    Weibull(α,θ)

The *Weibull distribution* with shape `α` and scale `θ` has probability density function

```math
f(x; \\alpha, \\theta) = \\frac{\\alpha}{\\theta} \\left( \\frac{x}{\\theta} \\right)^{\\alpha-1} e^{-(x/\\theta)^\\alpha},
    \\quad x \\ge 0
```

```julia
Weibull()        # Weibull distribution with unit shape and unit scale, i.e. Weibull(1, 1)
Weibull(α)       # Weibull distribution with shape α and unit scale, i.e. Weibull(α, 1)
Weibull(α, θ)    # Weibull distribution with shape α and scale θ

params(d)        # Get the parameters, i.e. (α, θ)
shape(d)         # Get the shape parameter, i.e. α
scale(d)         # Get the scale parameter, i.e. θ
```

External links

* [Weibull distribution on Wikipedia](http://en.wikipedia.org/wiki/Weibull_distribution)

"""
struct Weibull{T<:Real} <: ContinuousUnivariateDistribution
    α::T   # shape
    θ::T   # scale

    function Weibull{T}(α::T, θ::T) where {T <: Real}
        new{T}(α, θ)
    end
end

function Weibull(α::T, θ::T; check_args::Bool=true) where {T <: Real}
    @check_args Weibull (α, α > zero(α)) (θ, θ > zero(θ))
    return Weibull{T}(α, θ)
end

Weibull(α::Real, θ::Real; check_args::Bool=true) = Weibull(promote(α, θ)...; check_args=check_args)
Weibull(α::Integer, θ::Integer; check_args::Bool=true) = Weibull(float(α), float(θ); check_args=check_args)
Weibull(α::Real=1.0) = Weibull(α, one(α); check_args=false)

@distr_support Weibull 0.0 Inf

#### Conversions

convert(::Type{Weibull{T}}, α::Real, θ::Real) where {T<:Real} = Weibull(T(α), T(θ))
Base.convert(::Type{Weibull{T}}, d::Weibull) where {T<:Real} = Weibull{T}(T(d.α), T(d.θ))
Base.convert(::Type{Weibull{T}}, d::Weibull{T}) where {T<:Real} = d

#### Parameters

shape(d::Weibull) = d.α
scale(d::Weibull) = d.θ

params(d::Weibull) = (d.α, d.θ)
partype(::Weibull{T}) where {T<:Real} = T


#### Statistics

mean(d::Weibull) = d.θ * gamma(1 + 1/d.α)
median(d::Weibull) = d.θ * logtwo ^ (1/d.α)
mode(d::Weibull{T}) where {T<:Real} = d.α > 1 ? (iα = 1 / d.α; d.θ * (1 - iα)^iα) : zero(T)

var(d::Weibull) = d.θ^2 * gamma(1 + 2/d.α) - mean(d)^2

function skewness(d::Weibull)
    μ = mean(d)
    σ2 = var(d)
    σ = sqrt(σ2)
    r = μ / σ
    gamma(1 + 3/d.α) * (d.θ/σ)^3 - 3r - r^3
end

function kurtosis(d::Weibull)
    α, θ = params(d)
    μ = mean(d)
    σ = std(d)
    γ = skewness(d)
    r = μ / σ
    r2 = r^2
    r4 = r2^2
    (θ/σ)^4 * gamma(1 + 4/α) - 4γ*r - 6r2 - r4 - 3
end

function entropy(d::Weibull)
    α, θ = params(d)
    0.5772156649015328606 * (1 - 1/α) + log(θ/α) + 1
end


#### Evaluation

function pdf(d::Weibull, x::Real)
    α, θ = params(d)
    z = abs(x) / θ
    res = (α / θ) * z^(α - 1) * exp(-z^α)
    x < 0 || isinf(x) ? zero(res) : res
end

function logpdf(d::Weibull, x::Real)
    α, θ = params(d)
    z = abs(x) / θ
    res = log(α / θ) + xlogy(α - 1, z) - z^α
    x < 0 || isinf(x) ? oftype(res, -Inf) : res
end

zval(d::Weibull, x::Real) = (max(x, 0) / d.θ) ^ d.α
xval(d::Weibull, z::Real) = d.θ * z ^ (1 / d.α)

cdf(d::Weibull, x::Real) = -expm1(- zval(d, x))
ccdf(d::Weibull, x::Real) = exp(- zval(d, x))
logcdf(d::Weibull, x::Real) = log1mexp(- zval(d, x))
logccdf(d::Weibull, x::Real) = - zval(d, x)

quantile(d::Weibull, p::Real) = xval(d, -log1p(-p))
cquantile(d::Weibull, p::Real) = xval(d, -log(p))
invlogcdf(d::Weibull, lp::Real) = xval(d, -log1mexp(lp))
invlogccdf(d::Weibull, lp::Real) = xval(d, -lp)

function gradlogpdf(d::Weibull{T}, x::Real) where T<:Real
    if insupport(Weibull, x)
        α, θ = params(d)
        (α - 1) / x - α * x^(α - 1) / (θ^α)
    else
        zero(T)
    end
end


#### Sampling

rand(rng::AbstractRNG, d::Weibull) = xval(d, randexp(rng))

#### Fit model

"""
    fit_mle(::Type{<:Weibull}, x::AbstractArray{<:Real}; 
    alpha0::Real = 1, maxiter::Int = 1000, tol::Real = 1e-16)

Compute the maximum likelihood estimate of the [`Weibull`](@ref) distribution with Newton's method.
"""
function fit_mle(::Type{<:Weibull}, x::AbstractArray{<:Real};
    alpha0::Real = 1, maxiter::Int = 1000, tol::Real = 1e-16)

    N = 0

    lnx = map(log, x)
    lnxsq = lnx.^2
    mean_lnx = mean(lnx)

    # first iteration outside loop, prevents type instability in α, ϵ

    xpow0 = x.^alpha0
    sum_xpow0 = sum(xpow0)
    dot_xpowlnx0 = dot(xpow0, lnx)

    fx = dot_xpowlnx0 / sum_xpow0 - mean_lnx - 1 / alpha0
    ∂fx = (-dot_xpowlnx0^2 + sum_xpow0 * dot(lnxsq, xpow0)) / (sum_xpow0^2) + 1 / alpha0^2

    Δα = fx / ∂fx
    α = alpha0 - Δα

    ϵ = abs(Δα)
    N += 1

    while ϵ > tol && N < maxiter

        xpow = x.^α
        sum_xpow = sum(xpow)
        dot_xpowlnx = dot(xpow, lnx)

        fx = dot_xpowlnx / sum_xpow - mean_lnx - 1 / α
        ∂fx = (-dot_xpowlnx^2 + sum_xpow * dot(lnxsq, xpow)) / (sum_xpow^2) + 1 / α^2

        Δα = fx / ∂fx
        α -= Δα

        ϵ = abs(Δα)
        N += 1
    end

    θ = mean(x.^α)^(1 / α)
    return Weibull(α, θ)
end

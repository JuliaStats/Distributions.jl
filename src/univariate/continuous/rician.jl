"""
    Rician(ν, σ)

The *Rician distribution* with parameters `ν` and `σ` has probability density function:

```math
f(x; \\nu, \\sigma) = \\frac{x}{\\sigma^2} \\exp\\left( \\frac{-(x^2 + \\nu^2)}{2\\sigma^2} \\right) I_0\\left( \\frac{x\\nu}{\\sigma^2} \\right).
```

If shape and scale parameters `K` and `Ω` are given instead, `ν` and `σ` may be computed from them:

```math
\\sigma = \\sqrt{\\frac{\\Omega}{2(K + 1)}}, \\quad \\nu = \\sigma\\sqrt{2K}
```

```julia
Rician()         # Rician distribution with parameters ν=0 and σ=1
Rician(ν, σ)     # Rician distribution with parameters ν and σ

params(d)        # Get the parameters, i.e. (ν, σ)
shape(d)         # Get the shape parameter K = ν²/2σ²
scale(d)         # Get the scale parameter Ω = ν² + 2σ²
```

External links:

* [Rician distribution on Wikipedia](https://en.wikipedia.org/wiki/Rice_distribution)

"""
struct Rician{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    σ::T
    Rician{T}(ν, σ) where {T} = new{T}(ν, σ)
end

function Rician(ν::T, σ::T; check_args::Bool=true) where {T<:Real}
    @check_args Rician (ν, ν ≥ zero(ν)) (σ, σ ≥ zero(σ))
    return Rician{T}(ν, σ)
end

Rician() = Rician{Float64}(0.0, 1.0)
Rician(ν::Real, σ::Real; check_args::Bool=true) = Rician(promote(ν, σ)...; check_args=check_args)
Rician(ν::Integer, σ::Integer; check_args::Bool=true) = Rician(float(ν), float(σ); check_args=check_args)

@distr_support Rician 0.0 Inf

#### Conversions

function convert(::Type{Rician{T}}, ν::Real, σ::Real) where T<:Real
    Rician(T(ν), T(σ))
end

Base.convert(::Type{Rician{T}}, d::Rician) where {T<:Real} = Rician{T}(T(d.ν), T(d.σ))
Base.convert(::Type{Rician{T}}, d::Rician{T}) where {T<:Real} = d

#### Parameters

shape(d::Rician) = d.ν^2 / (2 * d.σ^2)
scale(d::Rician) = d.ν^2 + 2 * d.σ^2

params(d::Rician) = (d.ν, d.σ)
partype(d::Rician{T}) where {T<:Real} = T

#### Statistics

# helper
_Lhalf(x) = exp(x/2) * ((1-x) * besseli(zero(x), -x/2) - x * besseli(oneunit(x), -x/2))

mean(d::Rician) = d.σ * sqrthalfπ * _Lhalf(-d.ν^2/(2 * d.σ^2))
var(d::Rician) = 2 * d.σ^2 + d.ν^2 - halfπ * d.σ^2 * _Lhalf(-d.ν^2/(2 * d.σ^2))^2

function mode(d::Rician)
    m = mean(d)
    _minimize_gss(x -> -pdf(d, x), zero(m), m)
end

# helper: 1D minimization using Golden-section search
function _minimize_gss(f, a, b; tol=1e-12)
    ϕ = (√5 + 1) / 2
    c = b - (b - a) / ϕ
    d = a + (b - a) / ϕ
    while abs(b - a) > tol
        if f(c) < f(d)
            b = d
        else
            a = c
        end
        c = b - (b - a) / ϕ
        d = a + (b - a) / ϕ
    end
    (b + a) / 2
end

#### PDF/CDF/quantile delegated to NoncentralChisq

function quantile(d::Rician, x::Real)
    ν, σ = params(d)
    return sqrt(quantile(NoncentralChisq(2, (ν / σ)^2), x)) * σ
end

function cquantile(d::Rician, x::Real)
    ν, σ = params(d)
    return sqrt(cquantile(NoncentralChisq(2, (ν / σ)^2), x)) * σ
end

function pdf(d::Rician, x::Real)
    ν, σ = params(d)
    result = 2 * x / σ^2 * pdf(NoncentralChisq(2, (ν / σ)^2), (x / σ)^2)
    return x < 0 || isinf(x) ? zero(result) : result
end

function logpdf(d::Rician, x::Real)
    ν, σ = params(d)
    result = log(2 * abs(x) / σ^2) + logpdf(NoncentralChisq(2, (ν / σ)^2), (x / σ)^2)
    return x < 0 || isinf(x) ? oftype(result, -Inf) : result
end

function cdf(d::Rician, x::Real)
    ν, σ = params(d)
    result = cdf(NoncentralChisq(2, (ν / σ)^2), (x / σ)^2)
    return x < 0 ? zero(result) : result
end

function logcdf(d::Rician, x::Real)
    ν, σ = params(d)
    result = logcdf(NoncentralChisq(2, (ν / σ)^2), (x / σ)^2)
    return x < 0 ? oftype(result, -Inf) : result
end

#### Sampling

function rand(rng::AbstractRNG, d::Rician)
    x = randn(rng) * d.σ + d.ν
    y = randn(rng) * d.σ
    hypot(x, y)
end

#### Fitting

# implementation based on the Koay inversion technique
function fit(::Type{<:Rician}, x::AbstractArray{T}; tol=1e-12, maxiters=500) where T
    μ₁ = mean(x)
    μ₂ = var(x)
    r = μ₁ / √μ₂
    if r < sqrt(π/(4-π))
        ν = zero(float(T))
        σ = scale(fit(Rayleigh, x))
    else
        ξ(θ) = 2 + θ^2 - π/8 * exp(-θ^2 / 2) * ((2 + θ^2) * besseli(0, θ^2 / 4) + θ^2 * besseli(1, θ^2 / 4))^2
        g(θ) = sqrt(ξ(θ) * (1+r^2) - 2)
        θ = g(1)
        for j in 1:maxiters
            θ⁻ = θ
            θ = g(θ)
            abs(θ - θ⁻) < tol && break
        end
        ξθ = ξ(θ)
        σ = convert(float(T), sqrt(μ₂ / ξθ))
        ν = convert(float(T), sqrt(μ₁^2 + (ξθ - 2) * σ^2))
    end
    Rician(ν, σ)
end

# Not implemented:
#   skewness(d::Rician)
#   kurtosis(d::Rician)
#   entropy(d::Rician)
#   mgf(d::Rician, t::Real)
#   cf(d::Rician, t::Real)

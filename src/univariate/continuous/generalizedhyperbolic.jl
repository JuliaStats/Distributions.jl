export GeneralizedHyperbolic

raw"""
    GeneralizedHyperbolic(α, β, δ, μ=0, λ=1) # traditional
    GeneralizedHyperbolic(Val(:locscale), z, p=0, μ=0, σ=1, λ=1) # location-scale

The *generalized hyperbolic (GH) distribution* with traditional parameters:

- $\alpha>0$ (shape);
- $-\alpha<\beta<\alpha$ (skewness);
- $\delta>0$ ("scale", but not really, because it appears as an argument to the modified Bessel function of the 2nd kind in the normalizing constant);
- $\mu\in\mathbb R$ (location);
- $\lambda\in\mathbb R$ (shape, $\lambda\neq 1$ makes the distribution "generalized");

has probability density function:

```math
\frac{
 (\gamma/\delta)^{\lambda}
}{
 \sqrt{2\pi} K_{\lambda}(\delta \gamma)
}
e^{\beta (x-\mu)}
\frac{
 K_{\lambda-1/2}\left(\alpha\sqrt{\delta^2 + (x-\mu)^2}\right)
}{
 \left(\alpha^{-1} \sqrt{\delta^2 + (x-\mu)^2}\right)^{1/2 - \lambda}
}, \quad\gamma=\sqrt{\alpha^2 - \beta^2}
```

These paameters are actually stored in `struct GeneralizedHyperbolic{T<:Real}`.

The alternative location-scale parameterization is used in [2]. There, parameters have the following meaning:

- $z>0$ measures heavy-tailedness;
- $p\in\mathbb R$ measures skewness ($p=0$ results in a symmetric distribution);
- $\mu\in\mathbb R$ and $\sigma>0$ are location and scale;
- $\lambda\in\mathbb R$ is a shape parameter.

These parameters are _not_ stored in the struct. Use `params(d, Val(:locscale))`,
where `d` is an instance of `GeneralizedHyperbolic`, to retrieve them.

Advantages of this parameterization are:

- It's location-scale, whereas $\delta$ in the traditional parameterization isn't a true scale parameter.
- All parameters are either positive or unconstrained. The traditional parameterization has the complicated linear constraint $-\alpha<\beta<\alpha$.

External links:

1. [Generalized hyperbolic distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalised_hyperbolic_distribution).
2. Puig, Pedro, and Michael A. Stephens. “Goodness-of-Fit Tests for the Hyperbolic Distribution.” The Canadian Journal of Statistics / La Revue Canadienne de Statistique 29, no. 2 (2001): 309–20. https://doi.org/10.2307/3316079.
"""
struct GeneralizedHyperbolic{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    δ::T
    μ::T
    λ::T
    function GeneralizedHyperbolic{T}(α::T, β::T, δ::T, μ::T=zero(T), λ::T=one(T)) where T<:Real
        new{T}(α, β, δ, μ, λ)
    end
end

function GeneralizedHyperbolic(α::T, β::T, δ::T, μ::T=zero(T), λ::T=one(T); check_args::Bool=true) where T<:Real
	check_args && @check_args GeneralizedHyperbolic (α, α > zero(α)) (δ, δ > zero(δ)) (β, -α < β < α)
	GeneralizedHyperbolic{T}(α, β, δ, μ, λ)
end

GeneralizedHyperbolic(α::Real, β::Real, δ::Real, μ::Real=0, λ::Real=1; check_args::Bool=true) =
    GeneralizedHyperbolic(promote(α, β, δ, μ, λ)...; check_args)

GeneralizedHyperbolic(::Val{:locscale}, z::Real, p::Real=0, μ::Real=0, σ::Real=1, λ::Real=1; check_args::Bool=true) =
	GeneralizedHyperbolic(z * sqrt(1 + p^2)/σ, z * p / σ, σ, μ, λ; check_args)

params(d::GeneralizedHyperbolic) = (d.α, d.β, d.δ, d.μ, d.λ)
params(d::GeneralizedHyperbolic, ::Val{:locscale}) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    (; z=δ * γ, p=β / γ, μ, σ=δ, λ)
end
partype(::GeneralizedHyperbolic{T}) where T = T

minimum(::GeneralizedHyperbolic) = -Inf
maximum(::GeneralizedHyperbolic) = Inf
insupport(::GeneralizedHyperbolic, x::Real) = true

mean(d::GeneralizedHyperbolic) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)
    μ + β * δ / γ * besselk(1 + λ, δ * γ) / besselk(λ, δ * γ)
end

var(d::GeneralizedHyperbolic) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    t0 = besselk(0 + λ, δ * γ)
    t1 = besselk(1 + λ, δ * γ)
    t2 = besselk(2 + λ, δ * γ)
    δ / γ * t1/t0 - (β * δ / γ * t1/t0)^2 + (β * δ / γ)^2 * t2/t0
end

skewness(d::GeneralizedHyperbolic) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    t0 = besselk(0 + λ, δ * γ)
    t1 = besselk(1 + λ, δ * γ)
    t2 = besselk(2 + λ, δ * γ)
    t3 = besselk(3 + λ, δ * γ)
    (
        -3β * (δ / γ * t1/t0)^2 + 2 * (β * δ / γ * t1/t0)^3 + 3β * (δ / γ)^2 * t2/t0
        - 3 * (β * δ / γ)^3 * t1*t2/t0^2 + (β * δ / γ)^3 * t3/t0
    ) / sqrt(
        δ / γ * t1/t0 - (β * δ / γ * t1/t0)^2 + (β * δ / γ)^2 * t2/t0
    )^3
end

kurtosis(d::GeneralizedHyperbolic) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    t0 = besselk(0 + λ, δ * γ)
    t1 = besselk(1 + λ, δ * γ)
    t2 = besselk(2 + λ, δ * γ)
    t3 = besselk(3 + λ, δ * γ)
    t4 = besselk(4 + λ, δ * γ)
    (
        3 * γ^2 * t0^3 * t2 + 6 * β^2 * γ * δ * t0 * (t1^3 - 2t0 * t1 * t2 + t0^2 * t3)
        + β^4 * δ^2 * (-3 * t1^4 + 6t0 * t1^2 * t2 - 4 * t0^2 * t1 * t3 + t0^3 * t4)
    ) / (
        γ * t0 * t1 + β^2 * δ * (-t1^2 + t0 * t2)
    )^2 - 3 # EXCESS kurtosis
end

logpdf(d::GeneralizedHyperbolic, x::Real) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    (
        -0.5log(2π) - log(besselk(λ, γ * δ)) + λ * (log(γ) - log(δ))
        + β * (x - μ)
        + (λ - 1/2) * (0.5log(δ^2 + (x - μ)^2) - log(α))
        + log(besselk(λ - 1/2, α * sqrt(δ^2 + (x - μ)^2)))
    )
end

cdf(d::GeneralizedHyperbolic, x::Real) =
	if isinf(x)
		(x < 0) ? zero(x) : one(x)
	elseif isnan(x)
		typeof(x)(NaN)
	else
		quadgk(z -> pdf(d, z), -Inf, x, maxevals=10^4)[1]
	end

@quantile_newton GeneralizedHyperbolic

mgf(d::GeneralizedHyperbolic, t::Number) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    g = sqrt(α^2 - (t + β)^2)
    exp(t * μ) / g^λ * sqrt((α - β) * (α + β))^λ * besselk(λ, g * δ) / besselk(λ, γ * δ)
end

cf(d::GeneralizedHyperbolic, t::Number) = mgf(d, 1im * t)

rand(rng::AbstractRNG, d::GeneralizedHyperbolic) = begin
	α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    V = rand(rng, GeneralizedInverseGaussian(Val(:Wolfram), δ/γ, δ^2, λ))
    μ + β * V + sqrt(V) * randn(rng)
end
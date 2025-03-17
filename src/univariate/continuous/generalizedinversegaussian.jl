"""
    GeneralizedInverseGaussian(a, b, p)

The *generalized inverse Gaussian distribution* with parameters `a`, `b` and `p` has the probability density function

```math
f(x; \\lambda, \\psi, \\chi) = \\frac{(\\psi/\\chi)^{\\lambda/2}}{2 \\cdot \\operatorname{K}_{p}(\\sqrt{\\psi\\chi})} \\cdot x^{(\\lambda - 1)} \\cdot \\exp\\!( - \\frac{1}{2} (\\frac{\\chi}{x} + \\psi x)), \\quad x > 0
```

# TODO: Add julia examples of usage <29-04-23> 

Exernal links

* [Generalized Inverse Gaussian distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution)
"""
struct GeneralizedInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    λ::T
    ψ::T
    χ::T
    GeneralizedInverseGaussian{T}(λ::T, ψ::T, χ::T) where {T<:Real} = new{T}(λ, ψ, χ)
end

function GeneralizedInverseGaussian(λ::T, ψ::T, χ::T; check_args::Bool=true) where {T<:Real}
    @check_args GeneralizedInverseGaussian (ψ, ψ >= zero(ψ)) (χ, χ >= zero(χ))
    if iszero(χ)
        return Gamma(λ, 2 / ψ)
    elseif iszero(ψ)
        return InverseGamma(-λ, χ / 2)
    end
    return GeneralizedInverseGaussian{T}(λ, ψ, χ)
end

GeneralizedInverseGaussian(λ::Real, ψ::Real, χ::Real; check_args::Bool=true) = GeneralizedInverseGaussian(promote(λ, ψ, χ)...; check_args=check_args)
GeneralizedInverseGaussian(λ::Integer, ψ::Integer, χ::Integer; check_args::Bool=true) = GeneralizedInverseGaussian(float(λ), float(ψ), float(χ); check_args=check_args)

# TODO: Default distributions for parameters not supplied <29-04-23> 

@distr_support GeneralizedInverseGaussian 0.0 Inf

#### Conversions

function convert(::Type{GeneralizedInverseGaussian{T}}, λ::S, ψ::S, χ::S) where {T<:Real,S<:Real}
    GeneralizedInverseGaussian(T(λ), T(ψ), T(χ))
end
function Base.convert(::Type{GeneralizedInverseGaussian{T}}, d::GeneralizedInverseGaussian) where {T<:Real}
    GeneralizedInverseGaussian{T}(T(d.λ), T(d.ψ), T(d.χ))
end
Base.convert(::Type{GeneralizedInverseGaussian{T}}, d::GeneralizedInverseGaussian{T}) where {T<:Real} = d

#### Parameters

params(d::GeneralizedInverseGaussian) = (d.λ, d.ψ, d.χ)
partype(::GeneralizedInverseGaussian{T}) where {T} = T

#### Statistics

function mean(d::GeneralizedInverseGaussian)
    λ, ψ, χ = params(d)
    ω = sqrt(ψ * χ)
    η = sqrt(χ / ψ)
    return η * besselk(λ + 1, ω) / besselk(λ, ω)
end

function var(d::GeneralizedInverseGaussian)
    λ, ψ, χ = params(d)
    ω = sqrt(ψ * χ)
    return (χ / ψ) * (besselk(λ + 2, ω) / besselk(λ, ω) - (besselk(λ + 1, ω) / besselk(λ, ω))^2)
end

# TODO: skewness, kurtosis <29-04-23> 

function mode(d::GeneralizedInverseGaussian)
    λ, ψ, χ = params(d)
    ((λ - 1) + sqrt((λ - 1)^2 + ψ * χ)) / ψ
end

# NOTE: Entropy could be implemented for integer and half integer values of `p` using tabulated values of the derivative
# at  https://dlmf.nist.gov/10.38#E7 <04-05-23> 

#### Evaluation

function pdf(d::GeneralizedInverseGaussian{T}, x::Real) where {T<:Real}
    if x > 0
        λ, ψ, χ = params(d)
        ω = sqrt(ψ * χ)
        return (ψ / χ)^(λ / 2) / (2 * besselk(λ, ω)) * x^(λ - 1) * exp(-(ψ * x + χ / x) / 2)
    else
        return zero(T)
    end
end

# TODO: cdf <29-04-23> 

function logpdf(d::GeneralizedInverseGaussian{T}, x::Real) where {T<:Real}
    if x > 0
        λ, ψ, χ = params(d)
        return (λ / 2) * (log(ψ) - log(χ)) - logtwo - log(besselk(λ, sqrt(ψ * χ))) + (λ - 1) * log(x) - ψ * x / 2 - χ / (2 * x)
    else
        return -Inf
    end
end

function mgf(d::GeneralizedInverseGaussian{T}, t::Real) where {T<:Real}
    λ, ψ, χ = params(d)
    return (ψ / (ψ - 2t))^(λ / 2) * besselk(λ, sqrt(χ * (ψ - 2t))) / besselk(λ, sqrt(ψ * χ))
end

function cf(d::GeneralizedInverseGaussian{T}, t::Real) where {T<:Real}
    λ, ψ, χ = params(d)
    return (ψ / (ψ - 2(im * t)))^(λ / 2) * besselk(λ, sqrt(χ * (ψ - 2(im * t)))) / besselk(λ, sqrt(ψ * χ))
end

@quantile_newton GeneralizedInverseGaussian

#### Sampling

# TODO: rand <29-04-23> 

#### Fit model

# TODO: SufficientStats, suffstats, fit_mle <29-04-23> 

import SpecialFunctions: besselk

@doc raw"""
    GeneralizedInverseGaussian(μ, λ, θ)

The *generalized inverse Gaussian distribution* with parameters `\mu>0`, `\lambda>0` and real `\theta` has probability density function:

```math
f(x; \mu, \lambda, \theta) =
\frac{1}{
    2\mu^{\theta} K_{\theta}(\lambda/\mu)
} x^{\theta-1} \exp\left(
    -\frac{\lambda}{2} \left(\frac{1}{x} + \frac{x}{\mu^2}\right)
\right)
, \quad x > 0
```

External links:

* [Generalized Inverse Gaussian distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution).
* [`InverseGaussianDistribution` in Wolfram language](https://reference.wolfram.com/language/ref/InverseGaussianDistribution.html).
"""
struct GeneralizedInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    λ::T
    θ::T
    function GeneralizedInverseGaussian(μ::T, λ::T, θ::T; check_args::Bool=true) where T<:Real
        check_args && @check_args GeneralizedInverseGaussian (μ, μ > zero(μ)) (λ, λ > zero(λ))
        new{T}(μ, λ, θ)
    end
end

GeneralizedInverseGaussian(μ::Real, λ::Real, θ::Real=-1/2; check_args::Bool=true) =
    GeneralizedInverseGaussian(promote(μ, λ, θ)...; check_args)

@distr_support GeneralizedInverseGaussian 0.0 Inf

params(d::GeneralizedInverseGaussian) = (d.μ, d.λ, d.θ)
partype(::GeneralizedInverseGaussian{T}) where T = T

mode(d::GeneralizedInverseGaussian) = begin
    μ, λ, θ = params(d)
    tmp = μ * (θ - 1)
    μ / λ * (
        tmp + sqrt(λ^2 + tmp^2)
    )
end

mean(d::GeneralizedInverseGaussian) = begin
    μ, λ, θ = params(d)
    b0 = besselk(0 + θ, λ / μ)
    b1 = besselk(1 + θ, λ / μ)
    μ * b1 / b0
end

var(d::GeneralizedInverseGaussian) = begin
    μ, λ, θ = params(d)
    b0 = besselk(0 + θ, λ / μ)
    b1 = besselk(1 + θ, λ / μ)
    b2 = besselk(2 + θ, λ / μ)

    μ^2 * (b0 * b2 - b1^2) / b0^2
end

# Source: Wolfram
skewness(d::GeneralizedInverseGaussian) = begin
    μ, λ, θ = params(d)
    b0 = besselk(0 + θ, λ / μ)
    b1 = besselk(1 + θ, λ / μ)
    b2 = besselk(2 + θ, λ / μ)
    b3 = besselk(3 + θ, λ / μ)
    (
        2 * b1^3 - 3b0 * b1 * b2 + b0^2 * b3
    ) / sqrt(
        b0 * b2 - b1^2
    )^3
end

# Source: Wolfram
kurtosis(d::GeneralizedInverseGaussian) = begin
    μ, λ, θ = params(d)
    t0 = besselk(0 + θ, λ/μ)
    t1 = besselk(1 + θ, λ/μ)
    t2 = besselk(2 + θ, λ/μ)
    t3 = besselk(3 + θ, λ/μ)
    t4 = besselk(4 + θ, λ/μ)
    (
        -3 * t1^4 + 6t0 * t1^2 * t2 - 4 * t0^2 * t1 * t3 + t0^3 * t4
    ) / (
        t1^2 - t0 * t2
    )^2 - 3 # EXCESS kurtosis!
end

logpdf(d::GeneralizedInverseGaussian, x::Real) = begin
    μ, λ, θ = params(d)
    if x >= 0
        -log(2) - θ * log(μ) - log(besselk(θ, λ / μ)) - λ/2 * (1/x + x/μ^2) + (θ - 1) * log(x)
    else
        -Inf
    end
end

cdf(d::GeneralizedInverseGaussian, x::Real) =
    if isinf(x)
        (x < 0) ? zero(x) : one(x)
    elseif isnan(x)
        typeof(x)(NaN)
    else
        quadgk(z -> pdf(d, z), 0, x, maxevals=1000)[1]
    end

@quantile_newton GeneralizedInverseGaussian

mgf(d::GeneralizedInverseGaussian, t::Number) = begin
    μ, λ, θ = params(d)
    tmp = 1 - 2t * μ^2 / λ
    tmp^(-θ/2) * besselk(θ, λ/μ * sqrt(tmp)) / besselk(θ, λ/μ)
end

cf(d::GeneralizedInverseGaussian, t::Number) = mgf(d, 1im * t)

"""
    rand(rng::AbstractRNG, d::GeneralizedInverseGaussian)

Sample from the generalized inverse Gaussian distribution based on [1], end of Section 6.

### References

1. Devroye, Luc. 2014. “Random Variate Generation for the Generalized Inverse Gaussian Distribution.” Statistics and Computing 24 (2): 239–46. https://doi.org/10.1007/s11222-012-9367-z.
"""
rand(rng::AbstractRNG, d::GeneralizedInverseGaussian) = begin
    # If X ~ GIG(μ, λ, θ), then X/μ ~ _GIG(λ/μ, θ),
    # so mu * _GIG(λ/μ, θ) == GIG(μ, λ, θ)
    μ, λ, θ = params(d)
    μ * rand(rng, _GIG(λ/μ, θ))
end

# ===== Private two-parameter version =====
"""
    _GIG(λ, ω)

Two-parameter generalized inverse Gaussian distribution, only used for sampling.

If `X ~ GeneralizedInverseGaussian(μ, λ, θ)`, then `Y = X / μ` follows `_GIG(λ/μ, θ)`.
NOTE: the paper says (Section 1) that the second parameter of `_GIG` should be `ω = 2sqrt(b/a)`, but computations in Wolfram Mathematica show otherwise.

### References

1. Devroye, Luc. 2014. “Random Variate Generation for the Generalized Inverse Gaussian Distribution.” Statistics and Computing 24 (2): 239–46. https://doi.org/10.1007/s11222-012-9367-z.
"""
struct _GIG{T1<:Real, T2<:Real} <: ContinuousUnivariateDistribution
    ω::T1
    λ::T2
    function _GIG(ω::T1, λ::T2) where {T1<:Real, T2<:Real}
        @assert ω >= 0
        new{T1, T2}(ω, λ)
    end
end

params(d::_GIG) = (d.ω, d.λ)

logpdf(d::_GIG, x::Real) =
    if x >= 0
        -log(2 * besselk(d.λ, d.ω)) + (d.λ - 1) * log(x) - d.ω/2 * (x + 1/x)
    else
        -Inf
    end

"""
    rand(rng::AbstractRNG, d::_GIG)

Sampling from the _2-parameter_ generalized inverse Gaussian distribution based on [1], end of Section 6.

### References

1. Devroye, Luc. 2014. “Random Variate Generation for the Generalized Inverse Gaussian Distribution.” Statistics and Computing 24 (2): 239–46. https://doi.org/10.1007/s11222-012-9367-z.
"""
function rand(rng::AbstractRNG, d::_GIG{<:Real, <:Real})
    # ω, λ = params(d)
    # negative_lmb = λ < 0
    # λ = abs(λ)
    # FIXME: `λ = abs(λ)` causes the code to be type-unstable and leads to massive slowdown.
    # Benchmark: `@time rand(Distributions._GIG(.2,.7), 10^6);`
    # - with `λ = abs(λ)`:            5.85 seconds, 90.96 M allocations
    # - with `λ = abs(lmb)` as below: 0.41 seconds, 3 allocations

    ω, lmb = params(d)
    negative_lmb = lmb < 0
    λ = abs(lmb)

    α = sqrt(ω^2 + λ^2) - λ
    ψ(x) = -α * (cosh(x) - 1) - λ * (exp(x) - x - 1)
    ψprime(x) = -α * sinh(x) - λ * (exp(x) - 1)

    tmp = -ψ(1)
    t = if 0.5 <= tmp <= 2
        1.0
    elseif tmp > 2
        sqrt(2 / (α + λ))
    else
        log(4 / (α + 2λ))
    end

    tmp = -ψ(-1)
    s = if 0.5 <= tmp <= 2
        1.0
    elseif tmp > 2
        sqrt(4 / (α * cosh(1) + λ))
    else
        min(1/λ, log(1 + 1/α + sqrt(1 / α^2 + 2/α)))
    end

    eta, zeta, theta, xi = -ψ(t), -ψprime(t), -ψ(-s), ψprime(-s)
    p, r = 1/xi, 1/zeta

    t_ = t - r * eta
    s_ = s - p * theta
    q = t_ + s_

    chi(x) = if -s_ <= x <= t_
        1.0
    elseif x < -s_
        exp(-theta + xi * (x + s))
    else # x > t_
        exp(-eta - zeta * (x - t))
    end

    # Generation
    U, V, W = rand(rng), rand(rng), rand(rng)
    X = if U < q / (p + q + r)
        -s_ + q * V
    elseif U < (q + r) / (p + q + r)
        t_ - r * log(V)
    else
        -s_ + p * log(V)
    end
    while W * chi(X) > exp(ψ(X))
        U, V, W = rand(rng), rand(rng), rand(rng)
        X = if U < q / (p + q + r)
            -s_ + q * V
        elseif U < (q + r) / (p + q + r)
            t_ - r * log(V)
        else
            -s_ + p * log(V)
        end
    end

    tmp = λ/ω
    logGIG = log(tmp + sqrt(1 + tmp^2)) + X
    exp(ifelse(negative_lmb, -logGIG, logGIG))
end

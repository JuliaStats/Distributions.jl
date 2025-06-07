import SpecialFunctions: besselk

@doc raw"""
    GeneralizedInverseGaussian(a, b, p)

The *generalized inverse Gaussian distribution* with parameters `a>0`, `b>0` and real `p` has probability density function:

```math
f(x; a, b, p) =
\frac{(a/b)^(p/2)}{2 K_p(\sqrt{ab})}
x^{p-1} e^{-(ax + b/x)/2}, \quad x > 0
```

External links:

* [Generalized Inverse Gaussian distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution).
"""
struct GeneralizedInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    p::T
    function GeneralizedInverseGaussian{T}(a::T, b::T, p::T) where T<:Real
        new{T}(a, b, p)
    end
end

function GeneralizedInverseGaussian(a::T, b::T, p::T; check_args::Bool=true) where T<:Real
    check_args && @check_args GeneralizedInverseGaussian (a, a > zero(a)) (b, b > zero(b))
    GeneralizedInverseGaussian{T}(a, b, p)
end

GeneralizedInverseGaussian(a::Real, b::Real, p::Real; check_args::Bool=true) =
    GeneralizedInverseGaussian(promote(a, b, p)...; check_args)

"""
    GeneralizedInverseGaussian(::Val{:Wolfram}, μ::Real, λ::Real, θ::Real=-1/2)

Wolfram Language parameterization, equivalent to `InverseGamma(μ, λ)`. `μ, λ` must be positive.
Obtain parameters in Wolfram parameterization like `params(the_GIG, Val(:Wolfram))`
"""
GeneralizedInverseGaussian(::Val{:Wolfram}, μ::Real, λ::Real, θ::Real=-1/2; check_args::Bool=true) =
    GeneralizedInverseGaussian(λ / μ^2, λ, θ; check_args)

params(d::GeneralizedInverseGaussian) = (d.a, d.b, d.p)
params(d::GeneralizedInverseGaussian, ::Val{:Wolfram}) = (
    μ=sqrt(d.b/d.a), λ=d.b, θ=d.p
)
partype(::GeneralizedInverseGaussian{T}) where T = T

minimum(::GeneralizedInverseGaussian) = 0.0
maximum(::GeneralizedInverseGaussian) = Inf
insupport(::GeneralizedInverseGaussian, x::Real) = x >= 0

mode(d::GeneralizedInverseGaussian) = (
    (d.p - 1) + sqrt((d.p - 1)^2 + d.a * d.b)
) / d.a

mean(d::GeneralizedInverseGaussian) =
    sqrt(d.b/d.a) * besselk(d.p+1, sqrt(d.a*d.b)) / besselk(d.p, sqrt(d.a*d.b))

var(d::GeneralizedInverseGaussian) = begin
    tmp1 = sqrt(d.a * d.b)
    tmp2 = besselk(d.p, tmp1)
    d.b/d.a * (
        besselk(d.p+2, tmp1) / tmp2 - (besselk(d.p+1, tmp1) / tmp2)^2
    )
end

# Source: Wolfram
skewness(d::GeneralizedInverseGaussian) = begin
    μ, λ, θ = params(d, Val(:Wolfram))
    t0 = besselk(0 + θ, λ/μ)
    t1 = besselk(1 + θ, λ/μ)
    t2 = besselk(2 + θ, λ/μ)
    t3 = besselk(3 + θ, λ/μ)
    (
        2 * t1^3 - 3t0 * t1 * t2 + t0^2 * t3
    ) / sqrt(
        -t1^2 + t0 * t2
    )^3
end

# Source: Wolfram
kurtosis(d::GeneralizedInverseGaussian) = begin
    μ, λ, θ = params(d, Val(:Wolfram))
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

logpdf(d::GeneralizedInverseGaussian, x::Real) =
    if x >= 0
        (
            d.p / 2 * log(d.a / d.b) - log(2 * besselk(d.p, sqrt(d.a * d.b)))
            + (d.p - 1) * log(x) - (d.a * x + d.b / x) / 2
        )
    else
        -Inf
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
    a, b, p = params(d)
    sqrt(a / (a - 2t))^p * besselk(p, sqrt(b * (a - 2t))) / besselk(p, sqrt(a * b))
end

cf(d::GeneralizedInverseGaussian, t::Number) = mgf(d, 1im * t)

"""
    rand(rng::AbstractRNG, d::GeneralizedInverseGaussian)

Sample from the generalized inverse Gaussian distribution based on [1], end of Section 6.

### References

1. Devroye, Luc. 2014. “Random Variate Generation for the Generalized Inverse Gaussian Distribution.” Statistics and Computing 24 (2): 239–46. https://doi.org/10.1007/s11222-012-9367-z.
"""
rand(rng::AbstractRNG, d::GeneralizedInverseGaussian) = begin
    # Paper says ω = sqrt(b/a), but Wolfram disagrees
    ω = sqrt(d.a * d.b)
    sqrt(d.b / d.a) * rand(rng, _GIG(d.p, ω))
end

# ===== Private two-parameter version =====
"""
    _GIG(λ, ω)

Two-parameter generalized inverse Gaussian distribution, only used for sampling.

If `X ~ GeneralizedInverseGaussian(a, b, p)`, then `Y = sqrt(a/b) * X` follows `_GIG(p, 2sqrt(b * a))`.
NOTE: the paper says (Section 1) that the second parameter of `_GIG` should be `ω = 2sqrt(b/a)`, but computations in Wolfram Mathematica show otherwise.

### References

1. Devroye, Luc. 2014. “Random Variate Generation for the Generalized Inverse Gaussian Distribution.” Statistics and Computing 24 (2): 239–46. https://doi.org/10.1007/s11222-012-9367-z.
"""
struct _GIG{T1<:Real, T2<:Real} <: ContinuousUnivariateDistribution
    λ::T1
    ω::T2
    function _GIG(λ::T1, ω::T2) where {T1<:Real, T2<:Real}
        @assert ω >= 0
        new{T1, T2}(λ, ω)
    end
end

logpdf(d::_GIG, x::Real) =
    if x >= 0
        -log(2 * besselk(-d.λ, d.ω)) + (d.λ - 1) * log(x) - d.ω/2 * (x + 1/x)
    else
        -Inf
    end

"""
    rand(rng::AbstractRNG, d::_GIG)

Sampling from the _2-parameter_ generalized inverse Gaussian distribution based on [1], end of Section 6.

### References

1. Devroye, Luc. 2014. “Random Variate Generation for the Generalized Inverse Gaussian Distribution.” Statistics and Computing 24 (2): 239–46. https://doi.org/10.1007/s11222-012-9367-z.
"""
function rand(rng::AbstractRNG, d::_GIG)
    λ, ω = d.λ, d.ω
    (λ < 0) && return 1 / rand(rng, _GIG(-λ, ω))

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
    (tmp + sqrt(1 + tmp^2)) * exp(X)
end

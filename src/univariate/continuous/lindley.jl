"""
    Lindley(θ)

The one-parameter *Lindley distribution* with shape `θ > 0` has probability density
function

```math
f(x; \\theta) = \\frac{\\theta^2}{1 + \\theta} (1 + x) e^{-\\theta x}, \\quad x > 0
```

It was first described by Lindley[^1] and was studied in greater detail by Ghitany
et al.[^2]
Note that `Lindley(θ)` is a mixture of an `Exponential(θ)` and a `Gamma(2, θ)` with
respective mixing weights `p = θ/(1 + θ)` and `1 - p`.

[^1]: Lindley, D. V. (1958). Fiducial Distributions and Bayes' Theorem. Journal of the
      Royal Statistical Society: Series B (Methodological), 20(1), 102–107.
[^2]: Ghitany, M. E., Atieh, B., & Nadarajah, S. (2008). Lindley distribution and its
      application. Mathematics and Computers in Simulation, 78(4), 493–506.
"""
struct Lindley{T<:Real} <: ContinuousUnivariateDistribution
    θ::T

    Lindley{T}(θ::T) where {T} = new{T}(θ)
end

function Lindley(θ::Real; check_args::Bool=true)
    @check_args Lindley (θ, θ > zero(θ))
    return Lindley{typeof(θ)}(θ)
end

Lindley(θ::Integer; check_args::Bool=true) = Lindley(float(θ); check_args=check_args)

Lindley() = Lindley{Float64}(1.0)

Base.convert(::Type{Lindley{T}}, d::Lindley) where {T} = Lindley{T}(T(shape(d)))

@distr_support Lindley 0.0 Inf

### Parameters

shape(d::Lindley) = d.θ
params(d::Lindley) = (shape(d),)
partype(::Lindley{T}) where {T} = T

### Statistics

mean(d::Lindley) = (2 + d.θ) / d.θ / (1 + d.θ)

var(d::Lindley) = 2 / d.θ^2 - 1 / (1 + d.θ)^2

skewness(d::Lindley) = 2 * evalpoly(d.θ, (2, 6, 6, 1)) / evalpoly(d.θ, (2, 4, 1))^(3//2)

kurtosis(d::Lindley) = 3 * evalpoly(d.θ, (8, 32, 44, 24, 3)) / evalpoly(d.θ, (2, 4, 1))^2 - 3

mode(d::Lindley) = d.θ < 1 ? (1 - d.θ) / d.θ : zero(d.θ)

# Derived with Mathematica:
#     KLDivergence := ResourceFunction["KullbackLeiblerDivergence"]
#     KLDivergence[LindleyDistribution[θp], LindleyDistribution[θq]]
function kldivergence(p::Lindley, q::Lindley)
    θp = shape(p)
    θq = shape(q)
    a = (θp + 2) * (θp - θq) / θp / (1 + θp)
    b = 2 * log(θp) + log1p(θq) - 2 * log(θq) - log1p(θp)
    return b - a
end

# Derived with Mathematica based on https://mathematica.stackexchange.com/a/275765:
#     ShannonEntropy[dist_?DistributionParameterQ] :=
#         Expectation[-LogLikelihood[dist, {x}], Distributed[x, dist]]
#     Simplify[ShannonEntropy[LindleyDistribution[θ]]]
function entropy(d::Lindley)
    θ = shape(d)
    return 1 + exp(θ) * expinti(-θ) / (1 + θ) - 2 * log(θ) + log1p(θ)
end

### Evaluation

_lindley_mgf(θ, t) = θ^2 * (1 + θ - t) / (1 + θ) / (θ - t)^2

mgf(d::Lindley, t::Real) = _lindley_mgf(shape(d), t)

cf(d::Lindley, t::Real) = _lindley_mgf(shape(d), t * im)

cgf(d::Lindley, t::Real) = log1p(-t / (1 + d.θ)) - 2 * log1p(-t / d.θ)

_zero(d::Lindley, y::Real) = zero(shape(d)) * zero(y)
_oftype(d::Lindley, y::Real, x) = oftype(_zero(d, y), x)

function pdf(d::Lindley, y::Real)
    θ = shape(d)
    if isnan(y)
        return _oftype(d, y, NaN)
    elseif isfinite(y) && y > 0
        return θ^2 / (1 + θ) * (1 + y) * exp(-θ * y)
    else
        return _zero(d, y)
    end
end

function logpdf(d::Lindley, y::Real)
    θ = shape(d)
    if isnan(y)
        return _oftype(d, y, NaN)
    elseif isfinite(y) && y > 0
        return 2 * log(θ) - log1p(θ) + log1p(y) - θ * y
    else
        return _oftype(d, y, -Inf)
    end
end

function gradlogpdf(d::Lindley, y::Real)
    if isnan(y)
        return _oftype(d, y, NaN)
    elseif isfinite(y) && y > 0
        return inv(1 + y) - shape(d)
    else
        return _zero(d, y)
    end
end

function ccdf(d::Lindley, y::Real)
    θ = shape(d)
    θy = θ * y
    if isnan(y)
        return _oftype(d, y, NaN)
    elseif y > 0
        if isfinite(y)
            return (1 + θy / (1 + θ)) * exp(-θy)
        else
            return _zero(d, y)
        end
    else
        return _oftype(d, y, 1)
    end
end

function logccdf(d::Lindley, y::Real)
    θ = shape(d)
    if isnan(y)
        return _oftype(d, y, NaN)
    elseif y > 0
        if isfinite(y)
            return log1p(θ * (1 + y)) - log1p(θ) - θ * y
        else
            return _oftype(d, y, -Inf)
        end
    else
        return _zero(d, y)
    end
end

cdf(d::Lindley, y::Real) = 1 - ccdf(d, y)

logcdf(d::Lindley, y::Real) = log1p(-ccdf(d, y))

# Jodrá, P. (2010). Computer generation of random variables with Lindley or
# Poisson–Lindley distribution via the Lambert W function. Mathematics and Computers
# in Simulation, 81(4), 851–859.
function quantile(d::Lindley, q::Real)
    θ = shape(d)
    return -1 - inv(θ) - lambertw((1 + θ) * (q - 1) / exp(1 + θ), -1) / θ
end

### Sampling

# Ghitany, M. E., Atieh, B., & Nadarajah, S. (2008). Lindley distribution and its
# application. Mathematics and Computers in Simulation, 78(4), 493–506.
function rand(rng::AbstractRNG, d::Lindley{T}) where {T}
    θ = shape(d)
    λ = inv(θ)
    u = rand(rng, T)
    p = θ / (1 + θ)
    return T(rand(rng, u <= p ? Exponential{T}(λ) : Gamma{T}(2, λ)))
end

### Fitting

# Ghitany et al. (2008)
function fit_mle(::Type{<:Lindley}, x::AbstractArray{<:Real})
    x̄ = mean(x)
    return Lindley((1 - x̄ + sqrt((x̄ - 1)^2 + 8x̄)) / 2x̄)
end

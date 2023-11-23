"""
    GeneralizedExtremeValue(μ, σ, ξ)

The *Generalized extreme value distribution* with shape parameter `ξ`, scale `σ` and location `μ` has probability density function

```math
f(x; \\xi, \\sigma, \\mu) = \\begin{cases}
        \\frac{1}{\\sigma} \\left[ 1+\\left(\\frac{x-\\mu}{\\sigma}\\right)\\xi\\right]^{-1/\\xi-1} \\exp\\left\\{-\\left[ 1+ \\left(\\frac{x-\\mu}{\\sigma}\\right)\\xi\\right]^{-1/\\xi} \\right\\} & \\text{for } \\xi \\neq 0  \\\\
        \\frac{1}{\\sigma} \\exp\\left\\{-\\frac{x-\\mu}{\\sigma}\\right\\} \\exp\\left\\{-\\exp\\left[-\\frac{x-\\mu}{\\sigma}\\right]\\right\\} & \\text{for } \\xi = 0 \\\\
    \\end{cases}
```

for

```math
x \\in \\begin{cases}
        \\left[ \\mu - \\frac{\\sigma}{\\xi}, + \\infty \\right) & \\text{for } \\xi > 0 \\\\
        \\left( - \\infty, + \\infty \\right) & \\text{for } \\xi = 0 \\\\
        \\left( - \\infty, \\mu - \\frac{\\sigma}{\\xi} \\right] & \\text{for } \\xi < 0
    \\end{cases}
```

```julia
GeneralizedExtremeValue(μ, σ, ξ)      # Generalized Pareto distribution with shape ξ, scale σ and location μ.

params(d)       # Get the parameters, i.e. (μ, σ, ξ)
location(d)     # Get the location parameter, i.e. μ
scale(d)        # Get the scale parameter, i.e. σ
shape(d)        # Get the shape parameter, i.e. ξ (sometimes called c)
```

External links

* [Generalized extreme value distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution)

"""
struct GeneralizedExtremeValue{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ξ::T

    function GeneralizedExtremeValue{T}(μ::T, σ::T, ξ::T) where T
        σ > zero(σ) || error("Scale must be positive")
        new{T}(μ, σ, ξ)
    end
end

GeneralizedExtremeValue(μ::T, σ::T, ξ::T) where {T<:Real} = GeneralizedExtremeValue{T}(μ, σ, ξ)
GeneralizedExtremeValue(μ::Real, σ::Real, ξ::Real) = GeneralizedExtremeValue(promote(μ, σ, ξ)...)
function GeneralizedExtremeValue(μ::Integer, σ::Integer, ξ::Integer)
    return GeneralizedExtremeValue(float(μ), float(σ), float(ξ))
end

#### Conversions
function convert(::Type{GeneralizedExtremeValue{T}}, μ::Real, σ::Real, ξ::Real) where T<:Real
    GeneralizedExtremeValue(T(μ), T(σ), T(ξ))
end
function Base.convert(::Type{GeneralizedExtremeValue{T}}, d::GeneralizedExtremeValue) where {T<:Real}
    GeneralizedExtremeValue{T}(T(d.μ), T(d.σ), T(d.ξ))
end
Base.convert(::Type{GeneralizedExtremeValue{T}}, d::GeneralizedExtremeValue{T}) where {T<:Real} = d

minimum(d::GeneralizedExtremeValue{T}) where {T<:Real} =
        d.ξ > 0 ? d.μ - d.σ / d.ξ : -T(Inf)
maximum(d::GeneralizedExtremeValue{T}) where {T<:Real} =
        d.ξ < 0 ? d.μ - d.σ / d.ξ : T(Inf)


#### Parameters

shape(d::GeneralizedExtremeValue) = d.ξ
scale(d::GeneralizedExtremeValue) = d.σ
location(d::GeneralizedExtremeValue) = d.μ
params(d::GeneralizedExtremeValue) = (d.μ, d.σ, d.ξ)
@inline partype(d::GeneralizedExtremeValue{T}) where {T<:Real} = T


#### Statistics

testfd(d::GeneralizedExtremeValue) = d.ξ^3
g(d::GeneralizedExtremeValue, k::Real) = gamma(1 - k * d.ξ) # This should not be exported.

function median(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps() # ξ == 0
        return μ - σ * log(log(2))
    else
        return μ + σ * (log(2) ^ (- ξ) - 1) / ξ
    end
end

function mean(d::GeneralizedExtremeValue{T}) where T<:Real
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps(one(ξ)) # ξ == 0
        return μ + σ * MathConstants.γ
    elseif ξ < 1
        return μ + σ * (gamma(1 - ξ) - 1) / ξ
    else
        return T(Inf)
    end
end

function mode(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps(one(ξ)) # ξ == 0
        return μ
    else
        return μ + σ * ((1 + ξ) ^ (-ξ) - 1) / ξ
    end
end

function var(d::GeneralizedExtremeValue{T}) where T<:Real
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps(one(ξ)) # ξ == 0
        return σ ^2 * π^2 / 6
    elseif ξ < 1/2
        return σ^2 * (g(d, 2) - g(d, 1) ^ 2) / ξ^2
    else
        return T(Inf)
    end
end

function skewness(d::GeneralizedExtremeValue{T}) where T<:Real
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps(one(ξ)) # ξ == 0
        return 12sqrt(6) * zeta(3) / pi ^ 3 * one(T)
    elseif ξ < 1/3
        g1 = g(d, 1)
        g2 = g(d, 2)
        g3 = g(d, 3)
        return sign(ξ) * (g3 - 3g1 * g2 + 2g1^3) / (g2 - g1^2) ^ (3/2)
    else
        return T(Inf)
    end
end

function kurtosis(d::GeneralizedExtremeValue{T}) where T<:Real
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps(one(ξ)) # ξ == 0
        return T(12)/5
    elseif ξ < 1 / 4
        g1 = g(d, 1)
        g2 = g(d, 2)
        g3 = g(d, 3)
        g4 = g(d, 4)
        return (g4 - 4g1 * g3 + 6g2 * g1^2 - 3 * g1^4) / (g2 - g1^2)^2 - 3
    else
        return T(Inf)
    end
end

function entropy(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)
    return log(σ) + MathConstants.γ * ξ + (1 + MathConstants.γ)
end

function quantile(d::GeneralizedExtremeValue, p::Real)
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps(one(ξ)) # ξ == 0
        return μ + σ * (-log(-log(p)))
    else
        return μ + σ * ((-log(p))^(-ξ) - 1) / ξ
    end
end


#### Support

insupport(d::GeneralizedExtremeValue, x::Real) = minimum(d) <= x <= maximum(d)


#### Evaluation

function logpdf(d::GeneralizedExtremeValue{T}, x::Real) where T<:Real
    if x == -Inf || x == Inf || ! insupport(d, x)
      return -T(Inf)
    else
        (μ, σ, ξ) = params(d)

        z = (x - μ) / σ # Normalise x.
        if abs(ξ) < eps(one(ξ)) # ξ == 0
            t = z
            return -log(σ) - t - exp(-t)
        else
            if z * ξ == -1 # Otherwise, would compute zero to the power something.
                return -T(Inf)
            else
                t = (1 + z * ξ) ^ (-1/ξ)
                return - log(σ) + (ξ + 1) * log(t) - t
            end
        end
    end
end

function pdf(d::GeneralizedExtremeValue{T}, x::Real) where T<:Real
    if x == -Inf || x == Inf || ! insupport(d, x)
        return zero(T)
    else
        (μ, σ, ξ) = params(d)

        z = (x - μ) / σ # Normalise x.
        if abs(ξ) < eps(one(ξ)) # ξ == 0
            t = exp(-z)
            return (t * exp(-t)) / σ
        else
            if z * ξ == -1 # In this case: zero to the power something.
                return zero(T)
            else
                t = (1 + z*ξ)^(- 1 / ξ)
                return (t^(ξ + 1) * exp(- t)) / σ
            end
        end
    end
end

function logcdf(d::GeneralizedExtremeValue, x::Real)
    μ, σ, ξ = params(d)
    z = (x - μ) / σ
    return if abs(ξ) < eps(one(ξ)) # ξ == 0
        -exp(- z)
    else
        # y(x) = y(bound) = 0 if x is not in the support with lower/upper bound
        y = max(1 + z * ξ, 0)
        - y^(-1/ξ)
    end
end
cdf(d::GeneralizedExtremeValue, x::Real) = exp(logcdf(d, x))

ccdf(d::GeneralizedExtremeValue, x::Real) = - expm1(logcdf(d, x))
logccdf(d::GeneralizedExtremeValue, x::Real) = log1mexp(logcdf(d, x))


#### Sampling
function rand(rng::AbstractRNG, d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)

    # Generate a Float64 random number uniformly in (0,1].
    u = 1 - rand(rng)

    if abs(ξ) < eps(one(ξ)) # ξ == 0
        rd = - log(- log(u))
    else
        rd = expm1(- ξ * log(- log(u))) / ξ
    end

    return μ + σ * rd
end

"""
    GeneralizedPareto(μ, σ, ξ)

The *Generalized Pareto distribution* with shape parameter `ξ`, scale `σ` and location `μ` has probability density function

```math
f(x; \\mu, \\sigma, \\xi) = \\begin{cases}
        \\frac{1}{\\sigma}(1 + \\xi \\frac{x - \\mu}{\\sigma} )^{-\\frac{1}{\\xi} - 1} & \\text{for } \\xi \\neq 0 \\\\
        \\frac{1}{\\sigma} e^{-\\frac{\\left( x - \\mu \\right) }{\\sigma}} & \\text{for } \\xi = 0
    \\end{cases}~,
    \\quad x \\in \\begin{cases}
        \\left[ \\mu, \\infty \\right] & \\text{for } \\xi \\geq 0 \\\\
        \\left[ \\mu, \\mu - \\sigma / \\xi \\right] & \\text{for } \\xi < 0
    \\end{cases}
```

```julia
GeneralizedPareto()             # Generalized Pareto distribution with unit shape and unit scale, i.e. GeneralizedPareto(0, 1, 1)
GeneralizedPareto(k, s)         # Generalized Pareto distribution with shape k and scale s, i.e. GeneralizedPareto(0, k, s)
GeneralizedPareto(m, k, s)      # Generalized Pareto distribution with shape k, scale s and location m.

params(d)       # Get the parameters, i.e. (m, s, k)
location(d)     # Get the location parameter, i.e. m
scale(d)        # Get the scale parameter, i.e. s
shape(d)        # Get the shape parameter, i.e. k
```

External links

* [Generalized Pareto distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_Pareto_distribution)

"""
struct GeneralizedPareto{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ξ::T
    GeneralizedPareto{T}(μ::T, σ::T, ξ::T) where {T} = new{T}(μ, σ, ξ)
end

function GeneralizedPareto(μ::T, σ::T, ξ::T; check_args=true) where {T <: Real}
    check_args && @check_args(GeneralizedPareto, σ > zero(σ))
    return GeneralizedPareto{T}(μ, σ, ξ)
end

GeneralizedPareto(μ::Real, σ::Real, ξ::Real) = GeneralizedPareto(promote(μ, σ, ξ)...)

function GeneralizedPareto(μ::Integer, σ::Integer, ξ::Integer)
    GeneralizedPareto(float(μ), float(σ), float(ξ))
end
GeneralizedPareto(σ::T, ξ::Real) where {T <: Real} = GeneralizedPareto(zero(T), σ, ξ)
GeneralizedPareto() = GeneralizedPareto(0.0, 1.0, 1.0, check_args=false)

minimum(d::GeneralizedPareto) = d.μ
maximum(d::GeneralizedPareto{T}) where {T<:Real} = d.ξ < 0 ? d.μ - d.σ / d.ξ : Inf

#### Conversions
function convert(::Type{GeneralizedPareto{T}}, μ::S, σ::S, ξ::S) where {T <: Real, S <: Real}
    GeneralizedPareto(T(μ), T(σ), T(ξ))
end
function convert(::Type{GeneralizedPareto{T}}, d::GeneralizedPareto{S}) where {T <: Real, S <: Real}
    GeneralizedPareto(T(d.μ), T(d.σ), T(d.ξ), check_args=false)
end

#### Parameters

location(d::GeneralizedPareto) = d.μ
scale(d::GeneralizedPareto) = d.σ
shape(d::GeneralizedPareto) = d.ξ
params(d::GeneralizedPareto) = (d.μ, d.σ, d.ξ)
partype(::GeneralizedPareto{T}) where {T} = T

#### Statistics

median(d::GeneralizedPareto) = d.ξ == 0 ? d.μ + d.σ * logtwo : d.μ + d.σ * expm1(d.ξ * logtwo) / d.ξ

function mean(d::GeneralizedPareto{T}) where {T<:Real}
    if d.ξ < 1
        return d.μ + d.σ / (1 - d.ξ)
    else
        return T(Inf)
    end
end

function var(d::GeneralizedPareto{T}) where {T<:Real}
    if d.ξ < 0.5
        return d.σ^2 / ((1 - d.ξ)^2 * (1 - 2 * d.ξ))
    else
        return T(Inf)
    end
end

function skewness(d::GeneralizedPareto{T}) where {T<:Real}
    (μ, σ, ξ) = params(d)

    if ξ < (1/3)
        return 2(1 + ξ) * sqrt(1 - 2ξ) / (1 - 3ξ)
    else
        return T(Inf)
    end
end

function kurtosis(d::GeneralizedPareto{T}) where T<:Real
    (μ, σ, ξ) = params(d)

    if ξ < 0.25
        k1 = (1 - 2ξ) * (2ξ^2 + ξ + 3)
        k2 = (1 - 3ξ) * (1 - 4ξ)
        return 3k1 / k2 - 3
    else
        return T(Inf)
    end
end


#### Evaluation

function logpdf(d::GeneralizedPareto{T}, x::Real) where T<:Real
    (μ, σ, ξ) = params(d)

    # The logpdf is log(0) outside the support range.
    p = -T(Inf)

    if x >= μ
        z = (x - μ) / σ
        if abs(ξ) < eps()
            p = -z - log(σ)
        elseif ξ > 0 || (ξ < 0 && x < maximum(d))
            p = (-1 - 1 / ξ) * log1p(z * ξ) - log(σ)
        end
    end

    return p
end

pdf(d::GeneralizedPareto, x::Real) = exp(logpdf(d, x))

function logccdf(d::GeneralizedPareto{T}, x::Real) where T<:Real
    (μ, σ, ξ) = params(d)

    # The logccdf is log(0) outside the support range.
    p = -T(Inf)

    if x >= μ
        z = (x - μ) / σ
        if abs(ξ) < eps()
            p = -z
        elseif ξ > 0 || (ξ < 0 && x < maximum(d))
            p = (-1 / ξ) * log1p(z * ξ)
        end
    end

    return p
end

ccdf(d::GeneralizedPareto, x::Real) = exp(logccdf(d, x))
cdf(d::GeneralizedPareto, x::Real) = -expm1(logccdf(d, x))

function quantile(d::GeneralizedPareto{T}, p::Real) where T<:Real
    (μ, σ, ξ) = params(d)

    if p == 0
        z = zero(T)
    elseif p == 1
        z = ξ < 0 ? -1 / ξ : T(Inf)
    elseif 0 < p < 1
        if abs(ξ) < eps()
            z = -log1p(-p)
        else
            z = expm1(-ξ * log1p(-p)) / ξ
        end
    else
      z = T(NaN)
    end

    return μ + σ * z
end


#### Sampling

function rand(rng::AbstractRNG, d::GeneralizedPareto)
    # Generate a Float64 random number uniformly in (0,1].
    u = 1 - rand(rng)

    if abs(d.ξ) < eps()
        rd = -log(u)
    else
        rd = expm1(-d.ξ * log(u)) / d.ξ
    end

    return d.μ + d.σ * rd
end

"""
    GeneralizedPareto(μ, σ, ξ)

The *Generalized Pareto distribution* (GPD) with shape parameter `ξ`, scale `σ` and location `μ` has probability density function

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
GeneralizedPareto()             # GPD with unit shape and unit scale, i.e. GeneralizedPareto(0, 1, 1)
GeneralizedPareto(ξ)            # GPD with shape ξ and unit scale, i.e. GeneralizedPareto(0, 1, ξ)
GeneralizedPareto(σ, ξ)         # GPD with shape ξ and scale σ, i.e. GeneralizedPareto(0, σ, ξ)
GeneralizedPareto(μ, σ, ξ)      # GPD with shape ξ, scale σ and location μ.

params(d)       # Get the parameters, i.e. (μ, σ, ξ)
location(d)     # Get the location parameter, i.e. μ
scale(d)        # Get the scale parameter, i.e. σ
shape(d)        # Get the shape parameter, i.e. ξ
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

function GeneralizedPareto(μ::T, σ::T, ξ::T; check_args::Bool=true) where {T <: Real}
    @check_args GeneralizedPareto (σ, σ > zero(σ))
    return GeneralizedPareto{T}(μ, σ, ξ)
end

function GeneralizedPareto(μ::Real, σ::Real, ξ::Real; check_args::Bool=true)
    return GeneralizedPareto(promote(μ, σ, ξ)...; check_args=check_args)
end

function GeneralizedPareto(μ::Integer, σ::Integer, ξ::Integer; check_args::Bool=true)
    GeneralizedPareto(float(μ), float(σ), float(ξ); check_args=check_args)
end

function GeneralizedPareto(σ::Real, ξ::Real; check_args::Bool=true)
    GeneralizedPareto(zero(σ), σ, ξ; check_args=check_args)
end
function GeneralizedPareto(ξ::Real; check_args::Bool=true)
    GeneralizedPareto(zero(ξ), one(ξ), ξ; check_args=check_args)
end

GeneralizedPareto() = GeneralizedPareto{Float64}(0.0, 1.0, 1.0)

minimum(d::GeneralizedPareto) = d.μ
maximum(d::GeneralizedPareto{T}) where {T<:Real} = d.ξ < 0 ? d.μ - d.σ / d.ξ : Inf

#### Conversions
function convert(::Type{GeneralizedPareto{T}}, μ::S, σ::S, ξ::S) where {T <: Real, S <: Real}
    GeneralizedPareto(T(μ), T(σ), T(ξ))
end
function Base.convert(::Type{GeneralizedPareto{T}}, d::GeneralizedPareto) where {T<:Real}
    GeneralizedPareto{T}(T(d.μ), T(d.σ), T(d.ξ))
end
Base.convert(::Type{GeneralizedPareto{T}}, d::GeneralizedPareto{T}) where {T<:Real} = d

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

function logccdf(d::GeneralizedPareto, x::Real)
    μ, σ, ξ = params(d)
    z = max((x - μ) / σ, 0) # z(x) = z(μ) = 0 if x < μ (lower bound)
    return if abs(ξ) < eps(one(ξ)) # ξ == 0
        -z
    elseif ξ < 0
        # y(x) = y(μ - σ / ξ) = -1 if x > μ - σ / ξ (upper bound)
        -log1p(max(z * ξ, -1)) / ξ
    else
        -log1p(z * ξ) / ξ
    end
end
ccdf(d::GeneralizedPareto, x::Real) = exp(logccdf(d, x))

cdf(d::GeneralizedPareto, x::Real) = -expm1(logccdf(d, x))
logcdf(d::GeneralizedPareto, x::Real) = log1mexp(logccdf(d, x))

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

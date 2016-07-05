doc"""
    GeneralizedPareto(ξ, σ, μ)

The *Generalized Pareto distribution* with shape parameter `ξ`, scale `σ` and location `μ` has probability density function

$f(x; \xi, \sigma, \mu) = \begin{cases}
        \frac{1}{\sigma}(1 + \xi \frac{x - \mu}{\sigma} )^{-\frac{1}{\xi} - 1} & \text{for } \xi \neq 0 \\
        \frac{1}{\sigma} e^{-\frac{\left( x - \mu \right) }{\sigma}} & \text{for } \xi = 0
    \end{cases}~,
    \quad x \in \begin{cases}
        \left[ \mu, \infty \right] & \text{for } \xi \geq 0 \\
        \left[ \mu, \mu - \sigma / \xi \right] & \text{for } \xi < 0
    \end{cases}$


```julia
GeneralizedPareto()             # Generalized Pareto distribution with unit shape and unit scale, i.e. GeneralizedPareto(1, 1, 0)
GeneralizedPareto(k, s)         # Generalized Pareto distribution with shape k and scale s, i.e. GeneralizedPareto(k, s, 0)
GeneralizedPareto(k, s, m)      # Generalized Pareto distribution with shape k, scale s and location m.

params(d)       # Get the parameters, i.e. (k, s, m)
shape(d)        # Get the shape parameter, i.e. k
scale(d)        # Get the scale parameter, i.e. s
location(d)     # Get the location parameter, i.e. m
```

External links

* [Generalized Pareto distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_Pareto_distribution)

"""

immutable GeneralizedPareto{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ξ::T

    function GeneralizedPareto(μ::T, σ::T, ξ::T)
        @check_args(GeneralizedPareto, σ > zero(σ))
        new(μ, σ, ξ)
    end

end

GeneralizedPareto{T<:Real}(μ::T, σ::T, ξ::T) = GeneralizedPareto{T}(μ, σ, ξ)
GeneralizedPareto(μ::Real, σ::Real, ξ::Real) = GeneralizedPareto(promote(μ, σ, ξ)...)
function GeneralizedPareto(μ::Integer, σ::Integer, ξ::Integer)
    GeneralizedPareto(Float64(μ), Float64(σ), Float64(ξ))
end
GeneralizedPareto(ξ::Real, σ::Real) = GeneralizedPareto(0.0, σ, ξ)
GeneralizedPareto() = GeneralizedPareto(0.0, 1.0, 1.0)

minimum(d::GeneralizedPareto) = d.μ
maximum{T<:Real}(d::GeneralizedPareto{T}) = d.ξ < 0 ? d.μ - d.σ / d.ξ : Inf

#### Conversions
function convert{T <: Real, S <: Real}(::Type{GeneralizedPareto{T}}, μ::S, σ::S, ξ::S)
    GeneralizedPareto(T(μ), T(σ), T(ξ))
end
function convert{T <: Real, S <: Real}(::Type{GeneralizedPareto{T}}, d::GeneralizedPareto{S})
    GeneralizedPareto(T(d.μ), T(d.σ), T(d.ξ))
end

#### Parameters

location(d::GeneralizedPareto) = d.μ
scale(d::GeneralizedPareto) = d.σ
shape(d::GeneralizedPareto) = d.ξ
params(d::GeneralizedPareto) = (d.μ, d.σ, d.ξ)


#### Statistics

median(d::GeneralizedPareto) = d.μ + d.σ * expm1(d.ξ * log(2)) / d.ξ

function mean{T<:Real}(d::GeneralizedPareto{T})
    if d.ξ < 1
        return d.μ + d.σ / (1 - d.ξ)
    else
        return T(Inf)
    end
end

function var{T<:Real}(d::GeneralizedPareto{T})
    if d.ξ < 0.5
        return d.σ^2 / ((1 - d.ξ)^2 * (1 - 2 * d.ξ))
    else
        return T(Inf)
    end
end

function skewness{T<:Real}(d::GeneralizedPareto{T})
    (μ, σ, ξ) = params(d)

    if ξ < (1/3)
        return 2(1 + ξ) * sqrt(1 - 2ξ) / (1 - 3ξ)
    else
        return T(Inf)
    end
end

function kurtosis{T<:Real}(d::GeneralizedPareto{T})
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

function logpdf{T<:Real}(d::GeneralizedPareto{T}, x::Real)
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

function logccdf{T<:Real}(d::GeneralizedPareto{T}, x::Real)
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

function quantile{T<:Real}(d::GeneralizedPareto{T}, p::Real)
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

function rand(d::GeneralizedPareto)
    # Generate a Float64 random number uniformly in (0,1].
    u = 1 - rand()

    if abs(d.ξ) < eps()
        rd = -log(u)
    else
        rd = expm1(-d.ξ * log(u)) / d.ξ
    end

    return d.μ + d.σ * rd
end

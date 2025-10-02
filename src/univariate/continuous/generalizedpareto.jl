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
    nlog1pp = -log1p(-p * one(T))
    z = abs(ξ) < eps() ? nlog1pp : expm1(ξ * nlog1pp) / ξ
    return muladd(σ, z, μ)
end

#### Fitting

#
# MLE
#

"""
    GeneralizedParetoKnownMuTheta(μ, θ)

Represents a [`GeneralizedPareto`](@ref) where ``\\mu`` and ``\\theta=\\frac{\\xi}{\\sigma}`` are known.
"""
struct GeneralizedParetoKnownMuTheta{T} <: IncompleteDistribution
    μ::T
    θ::T
end
GeneralizedParetoKnownMuTheta(μ, θ) = GeneralizedParetoKnownMuTheta(promote(μ, θ)...)

struct GeneralizedParetoKnownMuThetaStats{T} <: SufficientStats
    μ::T  # known mean
    θ::T  # known theta
    ξ::T  # known shape
end
function GeneralizedParetoKnownMuThetaStats(μ, θ, ξ)
    return GeneralizedParetoKnownMuThetaStats(promote(μ, θ, ξ)...)
end

function suffstats(d::GeneralizedParetoKnownMuTheta, x::AbstractArray{<:Real})
    μ = d.μ
    θ = d.θ
    ξ = mean(xi -> log1p(θ * (xi - μ)), x) # mle estimate of ξ
    return GeneralizedParetoKnownMuThetaStats(μ, θ, ξ)
end

"""
    fit_mle(::Type{<:GeneralizedPareto}, x; μ, θ)

Compute the maximum likelihood estimate of the parameters of a [`GeneralizedPareto`](@ref)
where ``\\mu`` and ``\\theta=\\frac{\\xi}{\\sigma}`` are known.
"""
function fit_mle(::Type{<:GeneralizedPareto}, x::AbstractArray{<:Real}; μ::Real, θ::Real)
    return fit_mle(GeneralizedParetoKnownMuTheta(μ, θ), x)
end
function fit_mle(g::GeneralizedParetoKnownMuTheta, x::AbstractArray{<:Real})
    return fit_mle(g, suffstats(g, x))
end
function fit_mle(g::GeneralizedParetoKnownMuTheta, ss::GeneralizedParetoKnownMuThetaStats)
    ξ = ss.ξ
    return GeneralizedPareto(g.μ, ξ / g.θ, ξ)
end

#
# empirical bayes
#

"""
    GeneralizedParetoKnownMu(μ)

Represents a [`GeneralizedPareto`](@ref) where ``\\mu`` is known.
"""
struct GeneralizedParetoKnownMu{T} <: IncompleteDistribution
    μ::T
end

"""
    fit(::Type{<:GeneralizedPareto}, x; μ, kwargs...)

Fit a [`GeneralizedPareto`](@ref) with known location `μ` to the data `x`.

The fit is performed using the Empirical Bayes method of [^ZhangStephens2009][^Zhang2010].

# Keywords
- `sorted::Bool=issorted(x)`: If `true`, `x` is assumed to be sorted. If `false`, a sorted
    copy of `x` is made.
- `improved::Bool=true`: If `true`, use the adaptive empirical prior of [^Zhang2010].
    If `false`, use the simpler prior of [^ZhangStephens2009].
- `min_points::Int=30`: The minimum number of quadrature points to use when estimating the
    posterior mean of ``\\theta = \\frac{\\xi}{\\sigma}``.

[^ZhangStephens2009]: Jin Zhang & Michael A. Stephens (2009)
                      A New and Efficient Estimation Method for the Generalized Pareto Distribution,
                      Technometrics, 51:3, 316-325,
                      DOI: [10.1198/tech.2009.08017](https://doi.org/10.1198/tech.2009.08017)
[^Zhang2010]: Jin Zhang (2010) Improving on Estimation for the Generalized Pareto Distribution,
              Technometrics, 52:3, 335-339,
              DOI: [10.1198/TECH.2010.09206](https://doi.org/10.1198/TECH.2010.09206)
"""
function StatsBase.fit(::Type{<:GeneralizedPareto}, x::AbstractArray{<:Real}; μ::Real, kwargs...)
    return fit(GeneralizedParetoKnownMu(μ), x; kwargs...)
end
function StatsBase.fit(g::GeneralizedParetoKnownMu, x::AbstractArray{<:Real}; kwargs...)
    return fit_empiricalbayes(g, x; kwargs...)
end

# Note: our ξ is ZhangStephens2009's -k, and our θ is ZhangStephens2009's -θ

function fit_empiricalbayes(
    g::GeneralizedParetoKnownMu,
    x::AbstractArray{<:Real};
    sorted::Bool=issorted(vec(x)),
    improved::Bool=true,
    min_points::Int=30,
)
    μ = g.μ
    # fitting is faster when the data are sorted
    xsorted = sorted ? vec(x) : sort(vec(x))
    xmin = first(xsorted)
    xmax = last(xsorted)
    if xmin ≈ xmax
        # support is nearly a point. solution is not unique; any solution satisfying the
        # constraints σ/ξ ≈ 0 and ξ < 0 is acceptable. we choose the ξ = -1 solution, i.e.
        # the uniform distribution
        σ = xmax - μ
        return GeneralizedPareto(μ, max(eps(zero(σ)), σ), -1)
    end
    # empirical prior on θ
    n = length(xsorted)
    θ_prior = if improved
        _gpd_empirical_prior_improved(μ, xsorted, n)
    else
        _gpd_empirical_prior(μ, xsorted, n)
    end
    # estimate θ using empirical bayes
    θ_hat = _fit_gpd_θ_empirical_bayes(μ, xsorted, min_points, θ_prior)
    # estimate remaining parameters using MLE
    return fit_mle(GeneralizedParetoKnownMuTheta(μ, θ_hat), xsorted)
end

# estimate θ̂ = ∫θp(θ|x,μ)dθ, i.e. the posterior mean using quadrature over grid
# of minimum length `min_points + floor(sqrt(length(x)))` uniformly sampled over an
# empirical prior
function _fit_gpd_θ_empirical_bayes(μ, xsorted, min_points, θ_prior)
    n = length(xsorted)

    # quadrature points uniformly spaced on the quantiles of the θ prior
    npoints = min_points + floor(Int, sqrt(n))
    pmin = 1 // (2 * npoints)
    p = pmin:(1//npoints):(1 - pmin)
    θ = map(Base.Fix1(quantile, θ_prior), p)

    # estimate mean θ over the quadrature points
    # with weights as the normalized profile likelihood 
    lθ = map(θ -> _gpd_profile_loglikelihood(μ, θ, xsorted, n), θ)
    weights = softmax!(lθ)
    θ_hat = dot(weights, θ)

    return θ_hat
end

# Zhang & Stephens, 2009
function _gpd_empirical_prior(μ, xsorted, n=length(xsorted))
    xmax = xsorted[n]
    μ_star = -inv(xmax - μ)
    x_25 = xsorted[fld(n + 2, 4)]
    σ_star = inv(6 * (x_25 - μ))
    ξ_star = 1//2
    return GeneralizedPareto(μ_star, σ_star, ξ_star)
end

# Zhang, 2010
function _gpd_empirical_prior_improved(μ, xsorted, n=length(xsorted))
    xmax = xsorted[n]
    μ_star = (n - 1) / ((n + 1) * (μ - xmax))
    p = (3//10, 2//5, 1//2, 3//5, 7//10, 4//5, 9//10)  # 0.3:0.1:0.9
    q1 = (7, 6, 5, 4, 3, 2, 1)  # 10 .* (1 .- p)
    q2 = (91, 84, 75, 64, 51, 36, 19)  # 100 .* (1 .- p .^ 2)
    # q1/10- and q2/100- quantiles of xsorted without interpolation,
    # i.e. the α-quantile of x without interpolation is x[max(1, floor(Int, α * n + 1/2))]
    twon = 2 * n
    x1mp = map(qi -> xsorted[max(1, fld(qi * twon + 1, 20))], q1)
    x1mp2 = map(qi -> xsorted[max(1, fld(qi * twon + 1, 200))], q2)
    expkp = @. (x1mp2 - x1mp) / (x1mp - μ)
    σp = @. log(p, expkp) * (x1mp - μ) / (1 - expkp)
    σ_star = inv(2 * median(σp))
    ξ_star = 1
    return GeneralizedPareto(μ_star, σ_star, ξ_star)
end

# compute log joint likelihood p(x|μ,θ), with ξ the MLE given θ and x
function _gpd_profile_loglikelihood(μ, θ, x, n=length(x))
    d = fit_mle(GeneralizedParetoKnownMuTheta(μ, θ), x)
    return -n * (log(d.σ) + d.ξ + 1)
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

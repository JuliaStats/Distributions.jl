"""
    Tweedie(μ,σ,p)

The *Tweedie distribution* with mean `μ ≥ 0`, dispersion `σ ≥ 0` and power `1 ≥ p ≥ 2`.
When ``p = 1`` and ``\\sigma = 1`` it is equivalent to a quasi-Poisson distribution,
and when ``p = 2`` to the Gamma distribution. When ``1 > p > 2``, it is a compound
Poisson-Gamma distribution, with probability density function:

```math
f(x; \\mu, \\sigma, p) = \\frac{1}{x} W(x, \\sigma^2, p) exp \\left(
    \\frac{1}{\\sigma^2}} [ x \\frac{\\mu^(1-p)}{1-p} - \\frac{\\mu^(2-p)}{2-p} ]
    \\right), \\quad x > 0
```
where ``W`` is [Wright's generalized Bessel function](https://en.wikipedia.org/wiki/Bessel%E2%80%93Maitland_function).

Note that if ``1 > p > 2`` then the distribution is continuous with a point mass concentrated at zero.
If ``p = 1`` then the distribution is discrete.

Computation of [`pdf`](@ref) and [`logpdf`](@ref) is carried out using `Float64`.
Accuracy is generally higher than 1e-11, though for some parameter values it can
be as low as 1e-8.

```julia
Tweedie(μ, σ, p) # Tweedie distribution with location μ, scale σ and power p

params(d)        # Get the parameters, i.e. (μ, σ, p)
location(d)      # Get the location parameter, i.e. μ
scale(d)         # Get the scale parameter, i.e. σ
shape(d)         # Get the shape parameter, i.e. p

mean(d)          # Get the mean, i.e. μ
var(d)           # Get the variance, i.e. σ^2 * μ^p
```

External links

- [Tweedie distribution on Wikipedia](https://en.wikipedia.org/wiki/Tweedie_distribution)
- [Compound Poisson distribution on Wikipedia](https://en.wikipedia.org/wiki/Compound_Poisson_distribution)

References

- Dunn P. K., Smyth G. K. (2005). "Series evaluation of Tweedie exponential dispersion model densities"
  *Statistics and Computing* 15: 267–280.
"""
struct Tweedie{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    p::T

    Tweedie{T}(µ::T, σ::T, p::T) where {T<:Real} = new{T}(µ, σ, p)
end

function Tweedie(μ::T, σ::T, p::T; check_args::Bool=true) where {T <: Real}
    @check_args(
        Tweedie,
        (μ, μ >= 0),
        (σ, σ >= 0),
        (p, 1 <= p <= 2)
    )
    return Tweedie{T}(μ, σ, p)
end

#### Outer constructors
Tweedie(μ::Real, σ::Real, p::Real; check_args::Bool=true) =
    Tweedie(promote(μ, σ, p)...; check_args=check_args)
Tweedie(μ::Integer, σ::Integer, p::Integer; check_args::Bool=true) =
    Tweedie(float(μ), float(σ), float(p); check_args=check_args)

#### Conversions
convert(::Type{Tweedie{T}}, μ::S, σ::S, p::S) where {T <: Real, S <: Real} = Tweedie(T(μ), T(σ), T(p))
Base.convert(::Type{Tweedie{T}}, d::Tweedie) where {T<:Real} = Tweedie{T}(T(d.μ), T(d.σ), T(d.p))
Base.convert(::Type{Tweedie{T}}, d::Tweedie{T}) where {T<:Real} = d

@distr_support Tweedie 0 Inf

#### Parameters

params(d::Tweedie) = (d.μ, d.σ, d.p)
partype(::Tweedie{T}) where {T} = T

location(d::Tweedie) = d.μ
scale(d::Tweedie) = d.σ
shape(d::Tweedie) = d.p

Base.eltype(::Type{Tweedie{T}}) where {T} = float(T)

#### Statistics

mean(d::Tweedie) = float(d.μ)
var(d::Tweedie) = d.σ^2 * d.μ^d.p
std(d::Tweedie) = d.σ * d.μ^(d.p/2)

# Clark, David R. and Charles A. Thayer. 2004.
# “A Primer on the Exponential Family of Distributions.” CAS Discussion Paper Program, 117-148
# https://www.casact.org/sites/default/files/database/dpp_dpp04_04dpp117.pdf
skewness(d::Tweedie) = d.p * d.σ / sqrt(d.μ ^ (2 - d.p))
kurtosis(d::Tweedie) = d.p * (2 * d.p - 1) * d.σ^2 / d.μ ^ (2 - d.p)

function logpdf(d::Tweedie{T}, x::Real)::promote_type(eltype(d), typeof(x)) where {T <: Real}
    isnan(x) && return NaN
    x >= 0 || return -Inf
    # See: Dunn, Smyth (2005). "Series evaluation of Tweedie exponential dispersion model densities"
    # Statistics and Computing 15: 267–280.
    # pdf(y, μ, p, ϕ) = f(y, θ, ϕ) = c(y, ϕ) * exp(1/ϕ (y θ - κ(θ)))
    # κ = cumulant function
    # θ = function of expectation μ and power p
    # α = (2-p)/(1-p)
    # ϕ = σ^2
    # y = x
    # for 1<p<2:
    # c(y, ϕ) = 1/y * wrightbessel(a, b, z)
    # a = -α
    # b = 0
    # z = (p-1)^α/(2-p) / y^α / ϕ^(1-α)
    μ, p, ϕ = d.μ, d.p, d.σ^2
    if p == 1
        return logpdf(Poisson(μ / ϕ), x / ϕ)
    elseif p == 2
        return logpdf(Gamma(1 / ϕ, μ * ϕ), x)
    else
        θ = μ ^ (1 - p) / (1 - p)
        κ = μ ^ (2 - p) / (2 - p)
        α = (2 - p) / (1 - p)

        res = (x * θ - κ) / ϕ
        if x > 0
            z = ((p - 1) * ϕ / x) ^ α / ((2 - p) * ϕ)
            # Use log to reduce risks of overflow when p is close to 1
            wb = logwrightbessel(Float64(-α), 0.0, Float64(z))
            # Overflow in `logwrightbessel` doesn't generally indicate that the PDF
            # value would be larger than `typemax(Float64)`
            wb == Inf && return NaN
            res += wb - log(x)
        end
        return res
    end
end

function cdf(d::Tweedie, x::Real)::promote_type(eltype(d), typeof(x))
    isnan(x) && return NaN
    x == Inf && return 1
    x >= 0 || return 0
    μ = d.μ
    p = d.p
    ϕ = d.σ^2
    if p == 1
        return cdf(Poisson(μ / ϕ), x / ϕ)
    elseif p == 2
        return cdf(Gamma(1 / ϕ, μ * ϕ), x)
    else
        # the mass at zero has to be handled separately as `quadgk` never evaluates at bounds
        return pdf(d, 0) + quadgk(xi -> pdf(d, xi), 0, x, rtol=1e-12)[1]
    end
end

function rand(rng::AbstractRNG, d::Tweedie)
    μ, p, ϕ = d.μ, d.p, d.σ^2
    # note that sources often use β = 1/θ for Gamma distribution
    # e.g. https://en.wikipedia.org/wiki/Compound_Poisson_distribution
    if p == 1
        return ϕ * rand(rng, Poisson(μ / ϕ))
    elseif p == 2
        return rand(rng, Gamma(1 / ϕ, μ * ϕ))
    else
        λ = μ^(2 - p) / ((2 - p) * ϕ)
        α = (2 - p) / (1 - p)
        θ = ((p - 1) * ϕ) / μ^(1 - p)
        N = rand(rng, Poisson(λ))
        return N == 0 ? zero(θ) : rand(rng, Gamma(- N * α, θ))
    end
end

# Implementation inspired by `qtweedie` in R package tweedie
# licensed under MIT with authorization from Peter Dunn
function quantile(d::Tweedie{T}, q::Real)::eltype(d) where {T <: Real}
    μ, ϕ, p = d.μ, d.σ^2, d.p

    if q == 0
        return zero(T)
    elseif q == 1
        return convert(T, Inf)
    elseif q < 0 || q > 1
        throw(DomainError(q, "q must be between 0 and 1"))
    end

    if p == 1
        return ϕ * quantile(Poisson(μ / ϕ), q)
    elseif p == 2
        return quantile(Gamma(1 / ϕ, μ * ϕ), q)
    else
        # Handle point mass at zero
        p_zero = pdf(d, 0)
        if q <= p_zero
            return zero(T)
        end

        # Starting values via interpolation between Poisson and Gamma quantiles
        qp = ϕ * quantile(Poisson(μ / ϕ), q)
        qg = quantile(Gamma(1 / ϕ, μ * ϕ), q)
        startx = (qg - qp) * p + (2 * qp - qg)

        qstart = cdf(d, startx)
        rx = lx = startx
        if qstart == q
            return startx
        elseif qstart > q
            while true
                lx = lx / 2
                cdf(d, lx) < q && break
            end
        elseif qstart < q
            while true
                rx = 1.5 * (rx + 2)
                cdf(d, rx) > q && break
            end
        end

        # Cannot use `quantile_newton` as pdf is sometimes multimodal
        return quantile_bisect(d, q, lx, rx)
    end
end

function cquantile(d::Tweedie, q::Real)
    0 <= q <= 1 || throw(DomainError(q, "q must be between 0 and 1"))
    cq = 1.0 - q
    # Allow for 1 eps tolerance as due to the mass at zero
    # if `1 - q` is rounded up when storing in floating point,
    # `cquantile(d, ccdf(d, 0))` can be very different from zero,
    # which doesn't make mathematical sense
    if d.p < 2 && cq <= nextfloat(pdf(d, 0))
        return zero(eltype(d))
    else
        return quantile(d, cq)
    end
end

function invlogccdf(d::Tweedie, lp::Real)
    p = -expm1(lp)
    # Allow for 1 eps tolerance as due to the mass at zero
    # if `1 - q` is rounded up when storing in floating point,
    # `invlogccdf(d, logccdf(d, 0))` can be very different from zero,
    # which doesn't make mathematical sense
    if d.p < 2 && p <= nextfloat(pdf(d, 0))
        return zero(eltype(d))
    else
        return quantile(d, p)
    end
end
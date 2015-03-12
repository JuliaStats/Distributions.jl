# Raised Cosine distribution
#
# Ref: http://en.wikipedia.org/wiki/Raised_cosine_distribution
#

immutable Cosine <: ContinuousUnivariateDistribution
    μ::Float64
    s::Float64

    function Cosine(μ::Real, s::Real)
        s > 0.0 || error("s must be positive.")
        @compat new(Float64(μ), Float64(s))
    end

    @compat Cosine(μ::Real) = new(Float64(μ), 1.0)
    Cosine() = new(0.0, 1.0)
end

@distr_support Cosine d.μ - d.s d.μ + d.s


#### Parameters

location(d::Cosine) = d.μ
scale(d::Cosine) = d.s

params(d::Cosine) = (d.μ, d.s)


#### Statistics

mean(d::Cosine) = d.μ

median(d::Cosine) = d.μ

mode(d::Cosine) = d.μ

const _cosined_varcoef = 0.13069096604865779  # 1 / 3 - 2 / π^2
var(d::Cosine) = d.s^2 * _cosined_varcoef

skewness(d::Cosine) = 0.0

kurtosis(d::Cosine) = -0.59376287559828102362


#### Evaluation

function pdf(d::Cosine, x::Float64)
    if insupport(d, x)
        μ, s = params(d)
        z = (x - μ) / s
        return (1.0 + cospi(z)) / (2 * s)
    else
        return 0.0
    end
end

logpdf(d::Cosine, x::Float64) = insupport(d, x) ? log(pdf(d, x)) : -Inf

function cdf(d::Cosine, x::Float64)
    μ, s = params(d)
    z = (x - μ) / s
    0.5 * (1.0 + z + sinpi(z) * invπ)
end

function ccdf(d::Cosine, x::Float64)
    μ, s = params(d)
    nz = (μ - x) / s
    0.5 * (1.0 + nz + sinpi(nz) * invπ)
end

quantile(d::Cosine, p::Float64) = quantile_bisect(d, p)

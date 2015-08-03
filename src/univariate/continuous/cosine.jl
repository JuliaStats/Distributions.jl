# Raised Cosine distribution
#
# Ref: http://en.wikipedia.org/wiki/Raised_cosine_distribution
#

immutable Cosine <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64

    Cosine(μ::Real, σ::Real) = (@check_args(Cosine, σ > zero(σ)); new(μ, σ))
    Cosine(μ::Real) = new(μ, 1.0)
    Cosine() = new(0.0, 1.0)
end

@distr_support Cosine d.μ - d.σ d.μ + d.σ


#### Parameters

location(d::Cosine) = d.μ
scale(d::Cosine) = d.σ

params(d::Cosine) = (d.μ, d.σ)


#### Statistics

mean(d::Cosine) = d.μ

median(d::Cosine) = d.μ

mode(d::Cosine) = d.μ

var(d::Cosine) = d.σ^2 * 0.13069096604865779  # 0.130... = 1/3 - 2 / π^2

skewness(d::Cosine) = 0.0

kurtosis(d::Cosine) = -0.59376287559828102362


#### Evaluation

function pdf(d::Cosine, x::Float64)
    if insupport(d, x)
        z = (x - d.μ) / d.σ
        return (1.0 + cospi(z)) / (2 * d.σ)
    else
        return 0.0
    end
end

logpdf(d::Cosine, x::Float64) = insupport(d, x) ? log(pdf(d, x)) : -Inf

function cdf(d::Cosine, x::Float64)
    z = (x - d.μ) / d.σ
    0.5 * (1.0 + z + sinpi(z) * invπ)
end

function ccdf(d::Cosine, x::Float64)
    nz = (d.μ - x) / d.σ
    0.5 * (1.0 + nz + sinpi(nz) * invπ)
end

quantile(d::Cosine, p::Float64) = quantile_bisect(d, p)

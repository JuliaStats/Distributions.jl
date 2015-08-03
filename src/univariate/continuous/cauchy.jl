immutable Cauchy <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64

    Cauchy(μ::Real, σ::Real) = (@check_args(Cauchy, σ > zero(σ)); new(μ, σ))
    Cauchy(μ::Real) = new(μ, 1.0)
    Cauchy() = new(0.0, 1.0)
end

@distr_support Cauchy -Inf Inf

#### Parameters

location(d::Cauchy) = d.μ
scale(d::Cauchy) = d.σ

params(d::Cauchy) = (d.μ, d.σ)


#### Statistics

mean(d::Cauchy) = NaN
median(d::Cauchy) = d.μ
mode(d::Cauchy) = d.μ

var(d::Cauchy) = NaN
skewness(d::Cauchy) = NaN
kurtosis(d::Cauchy) = NaN

entropy(d::Cauchy) = log4π + log(d.σ)


#### Functions

zval(d::Cauchy, x::Float64) = (x - d.μ) / d.σ
xval(d::Cauchy, z::Float64) = d.μ + z * d.σ

pdf(d::Cauchy, x::Float64) = 1.0 / (π * scale(d) * (1.0 + zval(d, x)^2))
logpdf(d::Cauchy, x::Float64) = - (log1psq(zval(d, x)) + logπ + log(d.σ))

function cdf(d::Cauchy, x::Float64)
    μ, σ = params(d)
    invπ * atan2(x - μ, σ) + 0.5
end

function ccdf(d::Cauchy, x::Float64)
    μ, σ = params(d)
    invπ * atan2(μ - x, σ) + 0.5
end

function quantile(d::Cauchy, p::Float64)
    μ, σ = params(d)
    μ + σ * tan(π * (p - 0.5))
end

function cquantile(d::Cauchy, p::Float64)
    μ, σ = params(d)
    μ + σ * tan(π * (0.5 - p))
end

mgf(d::Cauchy, t::Real) = t == zero(t) ? 1.0 : NaN
cf(d::Cauchy, t::Real) = exp(im * (t * d.μ) - d.σ * abs(t))


#### Fitting

# Note: this is not a Maximum Likelihood estimator
function fit{T<:Real}(::Type{Cauchy}, x::AbstractArray{T})
    l, m, u = quantile(x, [0.25, 0.5, 0.75])
    Cauchy(m, (u - l) / 2.0)
end

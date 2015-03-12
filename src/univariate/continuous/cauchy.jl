immutable Cauchy <: ContinuousUnivariateDistribution
    μ::Float64
    β::Float64

    function Cauchy(μ::Real, β::Real)
        β > zero(β) || error("Cauchy: scale must be positive")
        @compat new(Float64(μ), Float64(β))
    end

    @compat Cauchy(μ::Real) = new(Float64(μ), 1.0)
    Cauchy() = new(0.0, 1.0)
end

@distr_support Cauchy -Inf Inf

#### Parameters

location(d::Cauchy) = d.μ
scale(d::Cauchy) = d.β

params(d::Cauchy) = (d.μ, d.β)


#### Statistics

mean(d::Cauchy) = NaN
median(d::Cauchy) = location(d)
mode(d::Cauchy) = location(d)

var(d::Cauchy) = NaN
skewness(d::Cauchy) = NaN
kurtosis(d::Cauchy) = NaN

entropy(d::Cauchy) = log(scale(d)) + log4π


#### Functions

zval(d::Cauchy, x::Float64) = (x - d.μ) / d.β
xval(d::Cauchy, z::Float64) = d.μ + z * d.β

pdf(d::Cauchy, x::Float64) = 1.0 / (π * scale(d) * (1 + zval(d, x)^2))
logpdf(d::Cauchy, x::Float64) = - (logπ + log(scale(d)) + log1psq(zval(d, x)))

function cdf(d::Cauchy, x::Float64)
    μ, β = params(d)
    invπ * atan2(x - μ, β) + 0.5
end

function ccdf(d::Cauchy, x::Float64)
    μ, β = params(d)
    invπ * atan2(μ - x, β) + 0.5
end

function quantile(d::Cauchy, p::Float64)
    μ, β = params(d)
    μ + β * tan(π * (p - 0.5))
end

function cquantile(d::Cauchy, p::Float64)
    μ, β = params(d)
    μ + β * tan(π * (0.5 - p))
end

mgf(d::Cauchy, t::Real) = t == zero(t) ? 1.0 : NaN
cf(d::Cauchy, t::Real) = exp(im * (t * d.μ) - d.β * abs(t))


#### Fitting

# Note: this is not a Maximum Likelihood estimator
function fit{T <: Real}(::Type{Cauchy}, x::Array{T})
    l, u = iqr(x)
    Cauchy(median(x), (u - l) / 2.0)
end

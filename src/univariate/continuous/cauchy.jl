immutable Cauchy <: ContinuousUnivariateDistribution
    x0::Float64
    γ::Float64

    function Cauchy(x0::Real, γ::Real)
        γ > zero(γ) || error("scale must be positive")
        new(float64(x0), float64(γ))
    end

    Cauchy(γ::Real) = Cauchy(0.0, γ)
    Cauchy() = new(0.0, 1.0)
end

@distr_support Cauchy -Inf Inf

#### Parameters

location(d::Cauchy) = d.x0
scale(d::Cauchy) = d.γ

params(d::Cauchy) = (d.x0, d.γ)


#### Statistics

mean(d::Cauchy) = NaN
median(d::Cauchy) = location(d)
mode(d::Cauchy) = location(d)

var(d::Cauchy) = NaN
skewness(d::Cauchy) = NaN
kurtosis(d::Cauchy) = NaN

entropy(d::Cauchy) = log(scale(d)) + log4π


#### Functions

function pdf(d::Cauchy, x::Float64)
    x0, γ = params(d)
    z = (x - x0) / γ
    1.0 / (π * γ * (1 + z^2))
end

function logpdf(d::Cauchy, x::Float64)
    x0, γ = params(d)
    z = (x - x0) / γ
    - (logπ + log(γ) + log1psq(z))
end

function cdf(d::Cauchy, x::Float64)
    x0, γ = params(d)
    invπ * atan2(x - x0, γ) + 0.5
end

function ccdf(d::Cauchy, x::Float64)
    x0, γ = params(d)
    invπ * atan2(x0 - x, γ) + 0.5
end

function quantile(d::Cauchy, p::Float64)
    x0, γ = params(d)
    x0 + γ * tan(π * (p - 0.5))
end

function cquantile(d::Cauchy, p::Float64)
    x0, γ = params(d)
    x0 + γ * tan(π * (0.5 - p))
end

mgf(d::Cauchy, t::Real) = t == zero(t) ? 1.0 : NaN

cf(d::Cauchy, t::Real) = exp(im * t * d.location - d.scale * abs(t))


#### Fitting

# Note: this is not a Maximum Likelihood estimator
function fit{T <: Real}(::Type{Cauchy}, x::Array{T})
    l, u = iqr(x)
    Cauchy(median(x), (u - l) / 2.0)
end

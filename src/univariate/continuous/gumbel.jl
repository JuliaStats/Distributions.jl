immutable Gumbel <: ContinuousUnivariateDistribution
    μ::Float64   # location
    β::Float64 # scale

    function Gumbel(μ::Real, β::Real)
        β > zero(β) || error("The scale of Gumbel must be positive")
        @compat new(Float64(μ), Float64(β))
    end

    Gumbel(μ::Real) = Gumbel(μ, 1.0)
    Gumbel() = new(0.0, 1.0)
end

@distr_support Gumbel -Inf Inf

const DoubleExponential = Gumbel


#### Parameters

location(d::Gumbel) = d.μ
scale(d::Gumbel) = d.β
params(d::Gumbel) = (d.μ, d.β)


#### Statistics

mean(d::Gumbel) = d.μ + d.β * 0.57721566490153286

median(d::Gumbel) = d.μ + d.β * 0.366512920581664327

mode(d::Gumbel) = d.μ

var(d::Gumbel) = 1.6449340668482264 * d.β^2

skewness(d::Gumbel) = 1.13954709940464866

kurtosis(d::Gumbel) = 2.4

entropy(d::Gumbel) = 1.57721566490153286 + log(d.β)


#### Evaluation

zval(d::Gumbel, x::Float64) = (x - d.μ) / d.β
xval(d::Gumbel, z::Float64) = x * d.β + d.μ

function pdf(d::Gumbel, x::Float64)
    z = zval(d, x)
    exp(-z - exp(-z)) / d.β
end

function logpdf(d::Gumbel, x::Float64)
    z = zval(d, x)
    - (z + exp(-z) + log(d.β))
end

cdf(d::Gumbel, x::Float64) = exp(-exp(-zval(d, x)))
logcdf(d::Gumbel, x::Float64) = -exp(-zval(d, x))

quantile(d::Gumbel, p::Float64) = d.μ - d.β * log(-log(p))

gradlogpdf(d::Gumbel, x::Float64) = - (1.0 + exp((d.μ - x) / d.β)) / d.β


#### Sampling

rand(d::Gumbel) = quantile(d, rand())



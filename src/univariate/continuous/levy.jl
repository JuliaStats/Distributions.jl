immutable Levy <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64

    Levy(μ::Real, σ::Real) = (@check_args(Levy, σ > zero(σ)); new(μ, σ))
    Levy(μ::Real) = new(μ, 1.0)
    Levy() = new(0.0, 1.0)
end

@distr_support Levy d.μ Inf


#### Parameters

location(d::Levy) = d.μ
params(d::Levy) = (d.μ, d.σ)


#### Statistics

mean(d::Levy) = Inf
var(d::Levy) = Inf
skewness(d::Levy) = NaN
kurtosis(d::Levy) = NaN

mode(d::Levy) = d.σ / 3.0 + d.μ

entropy(d::Levy) = (1.0 - 3.0 * digamma(1.0) + log(16π * d.σ^2)) / 2.0

median(d::Levy) = d.μ + d.σ / 0.4549364231195728  # 0.454... = (2.0 * erfcinv(0.5)^2)


#### Evaluation

function pdf(d::Levy, x::Float64)
    μ, σ = params(d)
    z = x - μ
    (sqrt(σ) / sqrt2π) * exp((-σ) / (2.0 * z)) / z^1.5
end

function logpdf(d::Levy, x::Float64)
    μ, σ = params(d)
    z = x - μ
    0.5 * (log(σ) - log2π - σ / z - 3.0 * log(z))
end

cdf(d::Levy, x::Float64) = erfc(sqrt(d.σ / (2.0 * (x - d.μ))))
ccdf(d::Levy, x::Float64) = erf(sqrt(d.σ / (2.0 * (x - d.μ))))

quantile(d::Levy, p::Float64) = d.μ + d.σ / (2.0 * erfcinv(p)^2)
cquantile(d::Levy, p::Float64) = d.μ + d.σ / (2.0 * erfinv(p)^2)

mgf(d::Levy, t::Real) = t == zero(t) ? 1.0 : NaN

function cf(d::Levy, t::Real)
    μ, σ = params(d)
    exp(im * μ * t - sqrt(-2.0 * im * σ * t))
end


#### Sampling

rand(d::Levy) = d.μ + d.σ / randn()^2

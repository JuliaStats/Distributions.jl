immutable Levy <: ContinuousUnivariateDistribution
    μ::Float64
    c::Float64

    function Levy(μ::Real, c::Real)
        c >= zero(c) || error("scale must be non-negative")
        @compat new(Float64(μ), Float64(c))
    end

    Levy(μ::Real) = new(μ, 1.0)
    Levy() = new(0.0, 1.0)
end

@distr_support Levy d.location Inf


#### Parameters

location(d::Levy) = d.μ
params(d::Levy) = (d.μ, d.c)


#### Statistics

mean(d::Levy) = Inf
var(d::Levy) = Inf
skewness(d::Levy) = NaN
kurtosis(d::Levy) = NaN

mode(d::Levy) = d.c / 3.0 + d.μ

function entropy(d::Levy)
    c = scale(d)
    (1.0 - 3.0 * digamma(1.0) + log(16.0 * pi * c * c)) / 2.0
end

function median(d::Levy)
    μ, c = params(d)
    μ + c / (2.0 * erfcinv(0.5)^2)
end


#### Evaluation

function pdf(d::Levy, x::Float64)
    μ, c = params(d)
    z = x - μ
    (sqrt(c) / sqrt2π) * exp((-c) / (2.0 * z)) / z^1.5
end

function logpdf(d::Levy, x::Float64)
    μ, c = params(d)
    z = x - μ
    0.5 * (log(c) - log2π - c / z - 3.0 * log(z))
end

cdf(d::Levy, x::Float64) = erfc(sqrt(d.c / (2.0 * (x - d.μ))))
ccdf(d::Levy, x::Float64) = erf(sqrt(d.c / (2.0 * (x - d.μ))))

quantile(d::Levy, p::Float64) = d.μ + d.c / (2.0 * erfcinv(p)^2)
cquantile(d::Levy, p::Float64) = d.μ + d.c / (2.0 * erfinv(p)^2)

mgf(d::Levy, t::Real) = t == zero(t) ? 1.0 : NaN

function cf(d::Levy, t::Real)
    μ, c = params(d)
    exp(im * μ * t - sqrt(-2.0 * im * c * t))
end


#### Sampling

rand(d::Levy) = d.μ + d.c / randn()^2


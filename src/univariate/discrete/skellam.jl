immutable Skellam <: DiscreteUnivariateDistribution
    μ1::Float64
    μ2::Float64

    function Skellam(μ1::Float64, μ2::Float64)
        μ1 > 0.0 && μ2 > 0.0 || error("μ1 and μ2 must be positive.")
        new(μ1, μ2)
    end

    Skellam(μ1::Real, μ2::Real) = Skellam(float64(μ1), float64(μ2))

    Skellam(μ::Real) = Skellam(μ, μ)

    Skellam() = new(1.0, 1.0)
end

@distr_support Skellam -Inf Inf


### Parameters

params(d::Skellam) = (d.μ1, d.μ2)


### Statistics

mean(d::Skellam) = d.μ1 - d.μ2

var(d::Skellam) = d.μ1 + d.μ2

skewness(d::Skellam) = mean(d) / (var(d)^1.5)

kurtosis(d::Skellam) = 1.0 / var(d)


### Evaluation

function logpdf(d::Skellam, x::Integer)
    μ1, μ2 = params(d)
    - (μ1 + μ2) + (x / 2.0) * log(μ1 / μ2) + log(besseli(x, 2.0 * sqrt(μ1) * sqrt(μ2)))
end

logpdf(d::Skellam, x::Real) = isinteger(x) ? logpdf(d, int(x)) : -Inf

pdf(d::Skellam, x::Integer) = exp(logpdf(d, x))

pdf(d::Skellam, x::Real) = isinteger(x) ? logpdf(d, int(x)) : 0.0

function mgf(d::Skellam, t::Real)
    μ1, μ2 = params(d)
    exp(-(μ1 + μ2) + μ1 * exp(t) + μ2 * exp(-t))
end

function cf(d::Skellam, t::Real)
    μ1, μ2 = params(d)
    exp(-(μ1 + μ2) + μ1 * exp(im * t) + μ2 * exp(im * -t))
end

### Sampling

rand(d::Skellam) = rand(Poisson(d.μ1)) - rand(Poisson(d.μ2))


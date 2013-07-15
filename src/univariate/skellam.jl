immutable Skellam <: DiscreteUnivariateDistribution
    mu1::Float64
    mu2::Float64
    Skellam(m1::Real, m2::Real) = new(float64(m1), float64(m2))
end

insupport(::Skellam, k::Real) = isinteger(k)
insupport(::Type{Skellam}, k::Real) = isinteger(k)

kurtosis(d::Skellam) = 1.0 / (d.mu1 + d.mu2)

mean(d::Skellam) = d.mu1 - d.mu2

function mgf(d::Skellam, t::Real)
    exp(-(d.mu1 + d.mu2) + d.mu1 * exp(t) + d.mu2 * exp(-t))
end

function cf(d::Skellam, t::Real)
    exp(-(d.mu1 + d.mu2) + d.mu1 * exp(im * t) + d.mu2 * exp(im * -t))
end

function pdf(d::Skellam, k::Real)
    isinteger(k) || return 0.0
    exp(-(d.mu1 + d.mu2)) * (d.mu1 / d.mu2)^(k / 2.0) *
        real(besseli(k, 2.0 * sqrt(d.mu1 * d.mu2)))
end

function logpdf(d::Skellam, k::Real)
    isinteger(k) || return 0.0
     -(d.mu1 + d.mu2) + (k / 2.0) * log(d.mu1) - log(d.mu2) +
           log(real(besseli(k, 2.0 * sqrt(d.mu1 * d.mu2))))
end

rand(d::Skellam) = rand(Poisson(d.mu1)) - rand(Poisson(d.mu2))

skewness(d::Skellam) = (d.mu1 - d.mu2) / (d.mu1 + d.mu2)^1.5

var(d::Skellam) = d.mu1 + d.mu2

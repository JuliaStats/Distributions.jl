immutable Laplace <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Laplace(l::Real, s::Real)
        if s > 0.0
            new(float64(l), float64(s))
        else
            error("scale must be positive")
        end
    end
end

Laplace(location::Real) = Laplace(location, 1.0)
Laplace() = Laplace(0.0, 1.0)

const Biexponential = Laplace

function cdf(d::Laplace, q::Real)
    return 0.5 * (1.0 + sign(q - d.location) *
           (1.0 - exp(-abs(q - d.location) / d.scale)))
end

entropy(d::Laplace) = log(2.0 * e * d.scale)

insupport(d::Laplace, x::Number) = isreal(x) && isfinite(x)

kurtosis(d::Laplace) = 3.0

mean(d::Laplace) = d.location

median(d::Laplace) = d.location

function mgf(d::Laplace, t::Real)
    m, b = d.location, d.scale
    return exp(t * m) / (1.0 - b^2 * t^2)
end

function cf(d::Laplace, t::Real)
    m, b = d.location, d.scale
    return exp(im * t * m) / (1.0 + b^2 * t^2)
end

modes(d::Laplace) = [d.location]

function pdf(d::Laplace, x::Real)
    (1.0 / (2.0 * d.scale)) * exp(-abs(x - d.location) / d.scale)
end

function logpdf(d::Laplace, x::Real)
    -log(2.0 * d.scale) - abs(x - d.location) / d.scale
end

function quantile(d::Laplace, p::Real)
    d.location - d.scale * sign(p - 0.5) * log(1.0 - 2.0 * abs(p - 0.5))
end

# Need to see whether other RNG strategies are more efficient:
# (1) Difference of two Exponential(1/b) variables
# (2) Ratio of logarithm of two Uniform(0.0, 1.0) variables
function rand(d::Laplace)
    u = rand() - 0.5
    return d.location - d.scale * sign(u) * log(1.0 - 2.0 * abs(u))
end

skewness(d::Laplace) = 0.0

std(d::Laplace) = sqrt(2.0) * d.scale

var(d::Laplace) = 2.0 * d.scale^2

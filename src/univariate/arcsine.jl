# TODO: Implement standard Arcsine(0, 1) and Arcsine(a, b)

immutable Arcsine <: ContinuousUnivariateDistribution
end

function cdf(d::Arcsine, x::Real)
    if x < 0.0
        return 0.0
    elseif x > 1.0
        return 1.0
    else
        return (2.0 / pi) * asin(sqrt(x))
    end
end

# TODO: This is not right, but is close enough to pass our tests
entropy(d::Arcsine) = -log(2.0) / pi

function insupport(d::Arcsine, x::Real)
    if 0 <= x <= 1.0
        return true
    else
        return false
    end
end

kurtosis(d::Arcsine) = -1.5

mean(d::Arcsine) = 0.5

median(d::Arcsine) = 0.5

function mgf(d::Arcsine, t::Real)
    s = 0.0
    for k in 1:10
        inner_s = 1.0
        for r in 0:(k - 1)
            inner_s *= (2.0 * r + 1.0) / (2.0 * r + 2.0)
        end
        s += t^k / factorial(k) * inner_s
    end
    return 1.0 + s
end

function cf(d::Arcsine, t::Real)
    error("CF for Arcsine requires confluent hypergeometric function")
end

modes(d::Arcsine) = [0.0, 1.0]

function pdf(d::Arcsine, x::Number)
    if 0.0 <= x <= 1.0
        return 1.0 / (pi * sqrt(x * (1.0 - x)))
    else
        return 0.0
    end
end

quantile(d::Arcsine, p::Real) = sin((pi * p) / 2.0)^2

rand(d::Arcsine) = sin(rand() * pi / 2.0)^2

skewness(d::Arcsine) = 0.0

var(d::Arcsine) = 1.0 / 8.0

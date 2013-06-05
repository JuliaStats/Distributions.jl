# TODO: Implement scaled Cosine and generalized Cosine
# TODO: This is all wrong. Replace entirely

immutable Cosine <: ContinuousUnivariateDistribution
end

rand(d::Cosine) = asin(2.0 * rand() - 1.0)

function cdf(d::Cosine, x::Real)
    if x < 0.0
        return 0.0
    elseif x > 1.0
        return 1.0
    else
        return 0.5 * (1 + sin(x))
    end
end

entropy(d::Cosine) = log(4.0 * pi) - 1.0

function insupport(d::Cosine, x::Real)
    if 0 <= x <= 1.0
        return true
    else
        return false
    end
end

kurtosis(d::Cosine) = -1.5

mean(d::Cosine) = 0.5

median(d::Cosine) = 0.5

# mgf(d::Cosine, t::Real)
# cf(d::Cosine, t::Real)

modes(d::Cosine) = [0.5]

function pdf(d::Cosine, x::Number)
    if 0.0 <= x <= 1.0
        return 0.5 * cos(x)
    else
        return 0.0
    end
end

quantile(d::Cosine, p::Real) = asin(2.0 * p - 1.0)

rand(d::Cosine) = sin(rand() * pi / 2.0)^2

skewness(d::Cosine) = 0.0

var(d::Cosine) = (pi^2 - 8.0) / (4.0 * pi^2)

# TODO: Implement scaled Cosine and generalized Cosine
# TODO: This is all wrong. Replace entirely

immutable Cosine <: ContinuousUnivariateDistribution
end

rand(d::Cosine) = asin(2.0 * rand() - 1.0)

function cdf(d::Cosine, x::Real)
    x < zero(x) ? 0.0 : (x > one(x) ? 1.0 : 0.5(1 + sin(x)))
end

entropy(d::Cosine) = log(4.0 * pi) - 1.0

insupport(::Cosine, x::Real) = zero(x) <= x <= one(x)
insupport(::Type{Cosine}, x::Real) = zero(x) <= x <= one(x)

kurtosis(d::Cosine) = -1.5

mean(d::Cosine) = 0.5

median(d::Cosine) = 0.5

# mgf(d::Cosine, t::Real)
# cf(d::Cosine, t::Real)

mode(d::Cosine) = 0.5
modes(d::Cosine) = [0.5]

pdf(d::Cosine, x::Real) = zero(x) <= x <= one(x) ? 0.5 * cos(x) : 0.0

quantile(d::Cosine, p::Real) = asin(2.p - 1.0)

rand(d::Cosine) = sin(rand() * pi / 2.0)^2

skewness(d::Cosine) = 0.0

var(d::Cosine) = (pi^2 - 8.0) / (4.0 * pi^2)

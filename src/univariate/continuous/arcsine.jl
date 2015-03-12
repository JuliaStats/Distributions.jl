immutable Arcsine <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64

    function Arcsine(a::Float64, b::Float64)
        a < b || error("a must be less than b.")
        new(a, b)
    end

    @compat Arcsine(a::Real, b::Real) = Arcsine(Float64(a), Float64(b))
    @compat Arcsine(b::Real) = Arcsine(0.0, Float64(b))
    Arcsine() = new(0.0, 1.0)
end

@distr_support Arcsine d.a d.b

### Parameters

params(d::Arcsine) = (d.a, d.b)
location(d::Arcsine) = d.a
scale(d::Arcsine) = d.b - d.a


### Statistics

mean(d::Arcsine) = (d.a + d.b) * 0.5
median(d::Arcsine) = mean(d)
mode(d::Arcsine) = d.a
modes(d::Arcsine) = [d.a, d.b]

var(d::Arcsine) = 0.125 * abs2(d.b - d.a)
skewness(d::Arcsine) = 0.0
kurtosis(d::Arcsine) = -1.5

entropy(d::Arcsine) = -0.24156447527049044469 + log(scale(d))


### Evaluation

pdf(d::Arcsine, x::Float64) = insupport(d, x) ? 1.0 / (π * sqrt((x - d.a) * (d.b - x))) : 0.0

logpdf(d::Arcsine, x::Float64) = insupport(d, x) ? -(logπ + 0.5 * log((x - d.a) * (d.b - x))) : -Inf

cdf(d::Arcsine, x::Float64) = x < d.a ? 0.0 :
                              x > d.b ? 1.0 :
                              0.636619772367581343 * asin(sqrt((x - d.a) / (d.b - d.a)))

quantile(d::Arcsine, p::Float64) = location(d) + abs2(sin(halfπ * p)) * scale(d)


### Sampling

rand(d::Arcsine) = quantile(d, rand())


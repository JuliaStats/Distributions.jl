doc"""
    Arcsine(a,b)

The *Arcsine distribution* has probability density function

$f(x) = \frac{1}{\pi \sqrt{(x - a) (b - x)}}, \quad x \in [a, b]$

```julia
Arcsine()        # Arcsine distribution with support [0, 1]
Arcsine(b)       # Arcsine distribution with support [0, b]
Arcsine(a, b)    # Arcsine distribution with support [a, b]

params(d)        # Get the parameters, i.e. (a, b)
minimum(d)       # Get the lower bound, i.e. a
maximum(d)       # Get the upper bound, i.e. b
location(d)      # Get the left bound, i.e. a
scale(d)         # Get the span of the support, i.e. b - a
```

External links

* [Arcsine distribution on Wikipedia](http://en.wikipedia.org/wiki/Arcsine_distribution)

"""
immutable Arcsine <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64

    Arcsine(a::Real, b::Real) = (@check_args(Arcsine, a < b); new(a, b))
    Arcsine(b::Real) = (@check_args(Arcsine, b > zero(b)); new(0.0, b))
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

# TODO: Implement standard Arcsine(0, 1) and Arcsine(a, b)

immutable Arcsine <: ContinuousUnivariateDistribution
end

function cdf(d::Arcsine, x::Real)
    x < zero(x) ? 0.0 : (x > one(x) ? 1.0 : (2.0 / pi) * asin(sqrt(x)))
end

# entropy(d::Arcsine) = log(pi) + digamma(0.5) + eulergamma
# calculated using higher-precision arithmetic 
entropy(d::Arcsine) = -0.24156447527049044469

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
    1.0 + s
end

function cf(d::Arcsine, t::Real)
    error("CF for Arcsine requires confluent hypergeometric function")
end

mode(d::Arcsine) = 0.0
modes(d::Arcsine) = [0.0, 1.0]

function pdf(d::Arcsine, x::Real)
    zero(x) <= x <= one(x) ? 1.0 / (pi * sqrt(x * (1.0 - x))) : 0.0
end

quantile(d::Arcsine, p::Real) = sin((pi * p) / 2.0)^2

rand(d::Arcsine) = sin(rand() * pi / 2.0)^2

skewness(d::Arcsine) = 0.0

var(d::Arcsine) = 1.0 / 8.0

### handling support

isupperbounded(::Union(Arcsine, Type{Arcsine})) = true
islowerbounded(::Union(Arcsine, Type{Arcsine})) = true
isbounded(::Union(Arcsine, Type{Arcsine})) = true

hasfinitesupport(::Union(Arcsine, Type{Arcsine})) = false
min(::Union(Arcsine, Type{Arcsine})) = zero(Real)
max(::Union(Arcsine, Type{Arcsine})) = one(Real)

insupport(::Union(Arcsine, Type{Arcsine}), x::Real) = min(Arcsine) <= x <= max(Arcsine)
"""
    Kumaraswamy(a, b)

The *Kumaraswamy distribution* with shape parameters `a > 0` and `b > 0` has probability
density function

```math
f(x; a, b) = a b x^{a - 1} (1 - x^a)^{b - 1}, \\quad 0 < x < 1
```

It is related to the [Beta distribution](@ref Beta) by the following identity:
if ``X \\sim \\operatorname{Kumaraswamy}(a, b)`` then ``X^a \\sim \\operatorname{Beta}(1, b)``.
In particular, if ``X \\sim \\operatorname{Kumaraswamy}(1, 1)`` then
``X \\sim \\operatorname{Uniform}(0, 1)``.

External links

- [Kumaraswamy distribution on Wikipedia](https://en.wikipedia.org/wiki/Kumaraswamy_distribution)

References

- Kumaraswamy, P. (1980). A generalized probability density function for double-bounded
  random processes. Journal of Hydrology. 46(1-2), 79-88.
"""
struct Kumaraswamy{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
end

function Kumaraswamy(a::Real, b::Real; check_args::Bool=true)
    @check_args Kumaraswamy (a, a > zero(a)) (b, b > zero(b))
    a′, b′ = promote(a, b)
    return Kumaraswamy{typeof(a′)}(a′, b′)
end

Kumaraswamy() = Kumaraswamy{Float64}(1.0, 1.0)

Base.convert(::Type{Kumaraswamy{T}}, d::Kumaraswamy) where {T} = Kumaraswamy{T}(T(d.a), T(d.b))
Base.convert(::Type{Kumaraswamy{T}}, d::Kumaraswamy{T}) where {T} = d

@distr_support Kumaraswamy 0 1

### Parameters

params(d::Kumaraswamy) = (d.a, d.b)
partype(::Kumaraswamy{T}) where {T} = T

### Evaluation

# `pdf`: Uses fallback `exp(logpdf(_))` method

function logpdf(d::Kumaraswamy, x::Real)
    a, b = params(d)
    _x = clamp(x, 0, 1)  # Ensures we can still get a value when outside the support
    y = log(a) + log(b) + xlogy(a - 1, _x) + xlog1py(b - 1, -_x^a)
    return x < 0 || x > 1 ? oftype(y, -Inf) : y
end

function ccdf(d::Kumaraswamy, x::Real)
    a, b = params(d)
    y = (1 - clamp(x, 0, 1)^a)^b
    return x < 0 ? one(y) : (x > 1 ? zero(y) : y)
end

cdf(d::Kumaraswamy, x::Real) = 1 - ccdf(d, x)

function logccdf(d::Kumaraswamy, x::Real)
    a, b = params(d)
    y = b * log1p(-clamp(x, 0, 1)^a)
    return x < 0 ? zero(y) : (x > 1 ? oftype(y, -Inf) : y)
end

logcdf(d::Kumaraswamy, x::Real) = log1mexp(logccdf(d, x))

function quantile(d::Kumaraswamy, q::Real)
    a, b = params(d)
    return (1 - (1 - q)^inv(b))^inv(a)
end

function entropy(d::Kumaraswamy)
    a, b = params(d)
    H = digamma(b + 1) + eulergamma
    return (1 - inv(b)) + (1 - inv(a)) * H - log(a) - log(b)
end

function gradlogpdf(d::Kumaraswamy, x::Real)
    a, b = params(d)
    _x = clamp(x, 0, 1)
    _xᵃ = _x^a
    y = (a * (b * _xᵃ - 1) + (1 - _xᵃ)) / (_x * (_xᵃ - 1))
    return x < 0 || x > 1 ? oftype(y, -Inf) : y
end

### Sampling

# `rand`: Uses fallback inversion sampling method

### Statistics

_kumomentaswamy(a, b, n) = b * beta(1 + n / a, b)

mean(d::Kumaraswamy) = _kumomentaswamy(params(d)..., 1)

function var(d::Kumaraswamy)
    a, b = params(d)
    m₁ = _kumomentaswamy(a, b, 1)
    m₂ = _kumomentaswamy(a, b, 2)
    return m₂ - m₁^2
end

function skewness(d::Kumaraswamy)
    a, b = params(d)
    μ = mean(d)
    σ² = var(d)
    m₂ = _kumomentaswamy(a, b, 2)
    m₃ = _kumomentaswamy(a, b, 3)
    return (2m₃ - μ * (3m₂ - μ^2)) / (σ² * sqrt(σ²))
end

function kurtosis(d::Kumaraswamy)
    a, b = params(d)
    μ = mean(d)
    m₂ = _kumomentaswamy(a, b, 2)
    m₃ = _kumomentaswamy(a, b, 3)
    m₄ = _kumomentaswamy(a, b, 4)
    return (m₄ + μ * (-4m₃ + μ * (6m₂ - 3μ^2))) / var(d)^2 - 3
end

function median(d::Kumaraswamy)
    a, b = params(d)
    return (1 - 2^-inv(b))^inv(a)
end

function mode(d::Kumaraswamy)
    a, b = params(d)
    m = ((a - 1) / (a * b - 1))^inv(a)
    return a >= 1 && b >= 1 && !(a == b == 1) ? m : oftype(m, NaN)
end

# Implementation based on chapters 2-4 of
# Arnold, Barry C., Narayanaswamy Balakrishnan, and Haikady Navada Nagaraja.
# A first course in order statistics. Society for Industrial and Applied Mathematics, 2008.

"""
    OrderStatistic{D<:UnivariateDistribution,S<:ValueSupport} <: UnivariateDistribution{S}

The distribution of an order statistic from IID samples from a univariate distribution.

    OrderStatistic(dist::UnivariateDistribution, n::Int, rank::Int; check_args::Bool=true)

Construct the distribution of the `rank` ``=i``th order statistic from `n` independent
samples from `dist`.

The ``i``th order statistic of a sample is the ``i``th element of the sorted sample.
For example, the 1st order statistic is the sample minimum, while the ``n``th order
statistic is the sample maximum.

If ``f`` is the probability density (mass) function of `dist` with distribution function
``F``, then the probability density function ``g`` of the order statistic for continuous
`dist` is
```math
g(x; n, i) = {n \\choose i} [F(x)]^{i-1} [1 - F(x)]^{n-i} f(x),
```
and the probability mass function ``g`` of the order statistic for discrete `dist` is
```math
g(x; n, i) = \\sum_{k=i}^n {n \\choose k} \\left( [F(x)]^k [1 - F(x)]^{n-k} - [F(x_-)]^k [1 - F(x_-)]^{n-k} \\right),
```
where ``x_-`` is the largest element in the support of `dist` less than ``x``.

For the joint distribution of a subset of order statistics, use
[`JointOrderStatistics`](@ref) instead.

## Examples

```julia
OrderStatistic(Cauchy(), 10, 1)              # distribution of the sample minimum
OrderStatistic(DiscreteUniform(10), 10, 10)  # distribution of the sample maximum
OrderStatistic(Gamma(1, 1), 11, 5)           # distribution of the sample median
```
"""
struct OrderStatistic{D<:UnivariateDistribution,S<:ValueSupport} <:
       UnivariateDistribution{S}
    dist::D
    n::Int
    rank::Int
    function OrderStatistic(
        dist::UnivariateDistribution, n::Int, rank::Int; check_args::Bool=true
    )
        @check_args(OrderStatistic, 1 ≤ rank ≤ n)
        return new{typeof(dist),value_support(typeof(dist))}(dist, n, rank)
    end
end

minimum(d::OrderStatistic) = minimum(d.dist)
maximum(d::OrderStatistic) = maximum(d.dist)
insupport(d::OrderStatistic, x::Real) = insupport(d.dist, x)

params(d::OrderStatistic) = tuple(params(d.dist)..., d.n, d.rank)
partype(d::OrderStatistic) = partype(d.dist)
Base.eltype(::Type{<:OrderStatistic{D}}) where {D} = Base.eltype(D)
Base.eltype(d::OrderStatistic) = eltype(d.dist)

# distribution of the ith order statistic from an IID uniform distribution, with CDF Uᵢₙ(x)
function _uniform_orderstatistic(d::OrderStatistic)
    n = d.n
    rank = d.rank
    return Beta{Int}(rank, n - rank + 1)
end

function logpdf(d::OrderStatistic, x::Real)
    b = _uniform_orderstatistic(d)
    p = cdf(d.dist, x)
    if value_support(typeof(d)) === Continuous
        return logpdf(b, p) + logpdf(d.dist, x)
    else
        return logdiffcdf(b, p, p - pdf(d.dist, x))
    end
end

for f in (:logcdf, :logccdf, :cdf, :ccdf)
    @eval begin
        function $f(d::OrderStatistic, x::Real)
            b = _uniform_orderstatistic(d)
            return $f(b, cdf(d.dist, x))
        end
    end
end

for f in (:quantile, :cquantile)
    @eval begin
        function $f(d::OrderStatistic, p::Real)
            # since cdf is Fᵢₙ(x) = Uᵢₙ(Fₓ(x)), and Uᵢₙ is invertible and increasing, we
            # have Fₓ(x) = Uᵢₙ⁻¹(Fᵢₙ(x)). then quantile function is
            # Qᵢₙ(p) = inf{x: p ≤ Fᵢₙ(x)} = inf{x: Uᵢₙ⁻¹(p) ≤ Fₓ(x)} = Qₓ(Uᵢₙ⁻¹(p))
            b = _uniform_orderstatistic(d)
            return quantile(d.dist, $f(b, p))
        end
    end
end

function rand(rng::AbstractRNG, d::OrderStatistic)
    # inverse transform sampling. Since quantile function is Qₓ(Uᵢₙ⁻¹(p)), we draw a random
    # variable from Uᵢₙ and pass it through the quantile function of `d.dist`
    T = eltype(d.dist)
    b = _uniform_orderstatistic(d)
    return T(quantile(d.dist, rand(rng, b)))
end

# Moments

## Uniform

mean(d::OrderStatistic{<:Uniform}) = d.rank * scale(d.dist) / (d.n + 1) + minimum(d)
std(d::OrderStatistic{<:Uniform}) = std(_uniform_orderstatistic(d)) * scale(d.dist)
var(d::OrderStatistic{<:Uniform}) = var(_uniform_orderstatistic(d)) * scale(d.dist)^2
skewness(d::OrderStatistic{<:Uniform}) = skewness(_uniform_orderstatistic(d))
kurtosis(d::OrderStatistic{<:Uniform}) = kurtosis(_uniform_orderstatistic(d))

## Exponential

function mean(d::OrderStatistic{<:Exponential})
    # Arnold, eq 4.6.6
    θ = scale(d.dist)
    T = float(typeof(one(θ)))
    return θ * _harmonicdiff(T, d.n - d.rank, d.n)
end
function var(d::OrderStatistic{<:Exponential})
    # Arnold, eq 4.6.7
    θ = scale(d.dist)
    T = float(typeof(one(θ)))
    return θ^2 * _polygamma_diff(T, 1, d.n + 1 - d.rank, d.n + 1)
end
function skewness(d::OrderStatistic{<:Exponential})
    θ = scale(d.dist)
    T = float(typeof(one(θ)))
    return -_polygamma_diff(T, 2, d.n + 1 - d.rank, d.n + 1) /
           _polygamma_diff(T, 1, d.n + 1 - d.rank, d.n + 1)^(3//2)
end
function kurtosis(d::OrderStatistic{<:Exponential})
    θ = scale(d.dist)
    T = float(typeof(one(θ)))
    return _polygamma_diff(T, 3, d.n + 1 - d.rank, d.n + 1) /
           _polygamma_diff(T, 1, d.n + 1 - d.rank, d.n + 1)^2
end
# Common utilities

_harmonicnum(T::Type{<:Real}, n::Int) = _harmonicnum_from(zero(T), 0, n)

function _harmonicnum_from(Hm::Real, m::Int, n::Int)
    # m ≤ n
    (n - m) < 25 && return sum(Base.Fix1(/, one(Hm)), (m + 1):n; init=Hm)
    return digamma(oftype(Hm, n + 1)) + Base.MathConstants.eulergamma
end

function _harmonicdiff(T::Type{<:Real}, m::Int, n::Int)
    # TODO: improve heuristic
    d = n - m
    m, n = minmax(m, n)
    abs(d) < 50 && return sign(d) * sum(Base.Fix1(/, one(T)), (m + 1):n; init=zero(T))
    Hm = _harmonicnum(T, m)
    Hn = _harmonicnum_from(Hm, m, n)
    return sign(d) * (Hn - Hm)
end

function _polygamma_from(m, ϕk::Real, k::Int, n::Int)
    # backwards recurrence is more stable than forwards
    gap = k - n
    gap > 10 || gap < 0 && return polygamma(m, oftype(ϕk, n))
    num = (-1)^(m + 1) * oftype(ϕk, factorial(m))
    f = Base.Fix1(/, num) ∘ Base.Fix2(^, m + 1)
    return sum(f, (k - 1):-1:n; init=ϕk)
end

function _polygamma_diff(T::Type{<:Real}, m::Int, k::Int, n::Int)
    d = n - k
    k, n = minmax(k, n)
    s = -sign(d)
    if abs(d) ≤ 10
        num = (-1)^m * s * T(factorial(m))
        f = Base.Fix1(/, num) ∘ Base.Fix2(^, m + 1)
        return sum(f, k:(n - 1); init=zero(T))
    end
    ϕn = polygamma(m, T(n))
    ϕk = _polygamma_from(m, ϕn, n, k)
    return s * (ϕn - ϕk)
end

function _polygamma_sum(T::Type{<:Real}, m::Int, k::Int, n::Int)
    k, n = minmax(k, n)
    ϕn = polygamma(m, T(n))
    ϕk = _polygamma_from(m, ϕn, n, k)
    return ϕn + ϕk
end

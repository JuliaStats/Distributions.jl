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

# Implementation based on chapters 2-4 of
# Arnold, Barry C., Narayanaswamy Balakrishnan, and Haikady Navada Nagaraja.
# A first course in order statistics. Society for Industrial and Applied Mathematics, 2008.

"""
    JointOrderStatistics <: ContinuousMultivariateDistribution

The joint distribution of a subset of order statistics from a sample from a continuous univariate distribution.

    JointOrderStatistics(
        dist::ContinuousUnivariateDistribution,
        n::Int,
        r::AbstractVector{Int}=1:n;
        check_args::Bool=true,
    )

Construct the joint distribution of order statistics `r` from an IID sample of size `n` from `dist`.

The ``i``th order statistic of a sample is the ``i``th element of the sorted sample.
For example, the 1st order statistic is the sample minimum, while the ``n``th order
statistic is the sample maximum.

`r` must be a sorted vector of unique integers between 1 and `n`.

## Examples

```julia
JointOrderStatistics(Normal(), 10)           # Product(fill(Normal(), 10)) restricted to ordered vectors
JointOrderStatistics(Cauchy(), 10, [1, 10])  # joint distribution of the extrema
```
"""
struct JointOrderStatistics{D<:ContinuousUnivariateDistribution,R<:AbstractVector{Int}} <:
       ContinuousMultivariateDistribution
    dist::D
    n::Int
    r::R
    function JointOrderStatistics(
        dist::ContinuousUnivariateDistribution,
        n::Int,
        r::AbstractVector{Int}=1:n;
        check_args::Bool=true,
    )
        @check_args(
            JointOrderStatistics,
            (n, n ≥ 1, "`n` must be a positive integer."),
            (
                r,
                1 ≤ first(r) && last(r) ≤ n && issorted(r) && allunique(r),
                "`r` must be a sorted vector of unique integers between 1 and `n`.",
            ),
        )
        return new{typeof(dist),typeof(r)}(dist, n, r)
    end
end

length(d::JointOrderStatistics) = length(d.r)
function insupport(d::JointOrderStatistics, x::AbstractVector)
    return length(d) == length(x) && issorted(x) && all(Base.Fix1(insupport, d.dist), x)
end
minimum(d::JointOrderStatistics) = fill(minimum(d.dist), length(d))
maximum(d::JointOrderStatistics) = fill(maximum(d.dist), length(d))

params(d::JointOrderStatistics) = tuple(params(d.dist)..., d.n, d.r)
partype(d::JointOrderStatistics) = partype(d.dist)
Base.eltype(::Type{<:JointOrderStatistics{D}}) where {D} = Base.eltype(D)

function logpdf(d::JointOrderStatistics, x::AbstractVector{<:Real})
    n = d.n
    r = d.r
    lp = sum(Base.Fix1(logpdf, d.dist), x)
    T = eltype(lp)
    lp += loggamma(T(n + 1))
    length(r) == n && return lp
    i = 0
    xᵢ = oftype(float(first(x)), -Inf)
    for (j, xⱼ) in zip(r, x)
        lp += _marginalize_range(d.dist, n, i, j, xᵢ, xⱼ, T)
        i = j
        xᵢ = xⱼ
    end
    j = n + 1
    xⱼ = oftype(xᵢ, Inf)
    lp += _marginalize_range(d.dist, n, i, j, xᵢ, xⱼ, T)
    return lp
end

# given ∏ₖf(xₖ), marginalize all xₖ for i < k < j
function _marginalize_range(dist, n, i, j, xᵢ, xⱼ, T)
    k = j - i - 1
    k == 0 && return zero(T)
    lpdiff = if i == 0
        logcdf(dist, xⱼ)
    elseif j == n + 1
        logccdf(dist, xᵢ)
    else
        logdiffcdf(dist, xⱼ, xᵢ)
    end
    return k * lpdiff - loggamma(T(k + 1))
end

function _rand!(rng::AbstractRNG, d::JointOrderStatistics, x::AbstractVector{<:Real})
    n = d.n
    if n == length(d.r)  # r == 1:n
        # direct method, slower than inversion method for large `n` and distributions with
        # fast quantile function or that use inversion sampling
        rand!(rng, d.dist, x)
        sort!(x)
    else
        # use exponential spacing method with inversion, where for gaps in the ranks, we
        # use the fact that the sum Y of k IID variables xₘ ~ Exp(1) is Y ~ Gamma(k, 1).
        # this is slow if length(d.r) is close to n and quantile for d.dist is expensive,
        # but this branch is probably taken when length(d.r) is small or much smaller than n.
        T = typeof(one(eltype(x)))
        s = zero(one(T))
        j = 0
        for (m, i) in zip(eachindex(x), d.r)
            k = j - i
            s += k > 1 ? rand(rng, Gamma(k, one(T); check_args=false)) : randexp(rng, T)
            j = i
            x[m] = s
        end
        k = n + 1 - j
        s += k > 1 ? rand(rng, Gamma(k, one(T); check_args=false)) : randexp(rng, T)
        x .= quantile.(d.dist, x ./ s)
    end
    return x
end

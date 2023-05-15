# Implementation based on chapters 2-4 of
# Arnold, Barry C., Narayanaswamy Balakrishnan, and Haikady Navada Nagaraja.
# A first course in order statistics. Society for Industrial and Applied Mathematics, 2008.

"""
    JointOrderStatistics <: ContinuousMultivariateDistribution

The joint distribution of a subset of order statistics from a sample from a continuous
univariate distribution.

    JointOrderStatistics(
        dist::ContinuousUnivariateDistribution,
        n::Int,
        ranks=Base.OneTo(n);
        check_args::Bool=true,
    )

Construct the joint distribution of order statistics for the specified `ranks` from an IID
sample of size `n` from `dist`.

The ``i``th order statistic of a sample is the ``i``th element of the sorted sample.
For example, the 1st order statistic is the sample minimum, while the ``n``th order
statistic is the sample maximum.

`ranks` must be a sorted vector or tuple of unique `Int`s between 1 and `n`.

For a single order statistic, use [`OrderStatistic`](@ref) instead.

## Examples

```julia
JointOrderStatistics(Normal(), 10)           # Product(fill(Normal(), 10)) restricted to ordered vectors
JointOrderStatistics(Cauchy(), 10, [1, 10])  # joint distribution of the extrema
```
"""
struct JointOrderStatistics{
    D<:ContinuousUnivariateDistribution,R<:Union{AbstractVector{Int},Tuple{Int,Vararg{Int}}}
} <: ContinuousMultivariateDistribution
    dist::D
    n::Int
    ranks::R
    function JointOrderStatistics(
        dist::ContinuousUnivariateDistribution,
        n::Int,
        ranks::Union{AbstractVector{Int},Tuple{Int,Vararg{Int}}}=Base.OneTo(n);
        check_args::Bool=true,
    )
        @check_args(
            JointOrderStatistics,
            (n, n ≥ 1, "`n` must be a positive integer."),
            (
                ranks,
                _are_ranks_valid(ranks, n),
                "`ranks` must be a sorted vector or tuple of unique integers between 1 and `n`.",
            ),
        )
        return new{typeof(dist),typeof(ranks)}(dist, n, ranks)
    end
end

_islesseq(x, y) = isless(x, y) || isequal(x, y)

function _are_ranks_valid(ranks, n)
    first(ranks) ≥ 1 || return false
    last(ranks) ≤ n || return false
    # combined with the above checks, this is equivalent to but faster than
    # issorted(ranks) && allunique(ranks)
    return issorted(ranks; lt=_islesseq)
end
function _are_ranks_valid(ranks::AbstractRange, n)
    first(ranks) ≥ 1 || return false
    last(ranks) ≤ n || return false
    return issorted(ranks) && allunique(ranks)
end

length(d::JointOrderStatistics) = length(d.ranks)
function insupport(d::JointOrderStatistics, x::AbstractVector)
    return length(d) == length(x) && issorted(x) && all(Base.Fix1(insupport, d.dist), x)
end
minimum(d::JointOrderStatistics) = Fill(minimum(d.dist), length(d))
maximum(d::JointOrderStatistics) = Fill(maximum(d.dist), length(d))

params(d::JointOrderStatistics) = tuple(params(d.dist)..., d.n, d.ranks)
partype(d::JointOrderStatistics) = partype(d.dist)
Base.eltype(::Type{<:JointOrderStatistics{D}}) where {D} = Base.eltype(D)
Base.eltype(d::JointOrderStatistics) = eltype(d.dist)

function logpdf(d::JointOrderStatistics, x::AbstractVector{<:Real})
    n = d.n
    ranks = d.ranks
    lp = loglikelihood(d.dist, x)
    T = typeof(lp)
    lp += loggamma(T(n + 1))
    length(ranks) == n && return lp
    i = first(ranks)
    xᵢ = first(x)
    if i > 1  # _marginalize_range(d.dist, 0, i, -Inf, xᵢ, T)
        lp += (i - 1) * logcdf(d.dist, xᵢ) - loggamma(T(i))
    end
    for (j, xⱼ) in Iterators.drop(zip(ranks, x), 1)
        lp += _marginalize_range(d.dist, i, j, xᵢ, xⱼ, T)
        i = j
        xᵢ = xⱼ
    end
    if i < n  # _marginalize_range(d.dist, i, n + 1, xᵢ, Inf, T)
        lp += (n - i) * logccdf(d.dist, xᵢ) - loggamma(T(n - i + 1))
    end
    return lp
end

# given ∏ₖf(xₖ), marginalize all xₖ for i < k < j
function _marginalize_range(dist, i, j, xᵢ, xⱼ, T)
    k = j - i - 1
    k == 0 && return zero(T)
    return k * T(logdiffcdf(dist, xⱼ, xᵢ)) - loggamma(T(k + 1))
end

function _rand!(rng::AbstractRNG, d::JointOrderStatistics, x::AbstractVector{<:Real})
    n = d.n
    if n == length(d.ranks)  # ranks == 1:n
        # direct method, slower than inversion method for large `n` and distributions with
        # fast quantile function or that use inversion sampling
        rand!(rng, d.dist, x)
        sort!(x)
    else
        # use exponential generation method with inversion, where for gaps in the ranks, we
        # use the fact that the sum Y of k IID variables xₘ ~ Exp(1) is Y ~ Gamma(k, 1).
        # Lurie, D., and H. O. Hartley. "Machine-generation of order statistics for Monte
        # Carlo computations." The American Statistician 26.1 (1972): 26-27.
        # this is slow if length(d.ranks) is close to n and quantile for d.dist is expensive,
        # but this branch is probably taken when length(d.ranks) is small or much smaller than n.
        T = typeof(one(eltype(x)))
        s = zero(eltype(x))
        i = 0
        for (m, j) in zip(eachindex(x), d.ranks)
            k = j - i
            s += k > 1 ? rand(rng, Gamma(k, one(T); check_args=false)) : randexp(rng, T)
            i = j
            x[m] = s
        end
        j = n + 1
        k = j - i
        s += k > 1 ? rand(rng, Gamma(k, one(T); check_args=false)) : randexp(rng, T)
        x .= quantile.(d.dist, x ./ s)
    end
    return x
end

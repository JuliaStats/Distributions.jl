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
JointOrderStatistics(Cauchy(), 10, 2:9)      # joint distribution of all but the extrema
JointOrderStatistics(Cauchy(), 10, (1, 10))  # joint distribution of only the extrema
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
    # this is equivalent to but faster than
    # issorted(ranks) && allunique(ranks)
    !isempty(ranks) && first(ranks) ≥ 1 && last(ranks) ≤ n && issorted(ranks; lt=_islesseq)
end
function _are_ranks_valid(ranks::AbstractRange, n)
    !isempty(ranks) && first(ranks) ≥ 1 && last(ranks) ≤ n && step(ranks) > 0
end

length(d::JointOrderStatistics) = length(d.ranks)
function insupport(d::JointOrderStatistics, x::AbstractVector)
    length(d) == length(x) || return false
    xi, state = iterate(x) # at least one element!
    dist = d.dist
    insupport(dist, xi) || return false
    while (xj_state = iterate(x, state)) !== nothing
        xj, state = xj_state
        xj ≥ xi && insupport(dist, xj) || return false
        xi = xj
    end
    return true
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
    if length(ranks) == n
        issorted(x) && return lp
        return oftype(lp, -Inf)
    end
    i = first(ranks)
    xᵢ = first(x)
    if i > 1  # _marginalize_range(d.dist, 0, i, -Inf, xᵢ, T)
        lp += (i - 1) * logcdf(d.dist, xᵢ) - loggamma(T(i))
    end
    for (j, xⱼ) in Iterators.drop(zip(ranks, x), 1)
        xⱼ < xᵢ && return oftype(lp, -Inf)
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
            if k > 1
                # specify GammaMTSampler directly to avoid unnecessarily checking the shape
                # parameter again and because it has been benchmarked to be the fastest for
                # shape k ≥ 1 and scale 1
                s += T(rand(rng, GammaMTSampler(Gamma{T}(T(k), T(1)))))
            else
                s += randexp(rng, T)
            end
            i = j
            x[m] = s
        end
        j = n + 1
        k = j - i
        if k > 1
            s += T(rand(rng, GammaMTSampler(Gamma{T}(T(k), T(1)))))
        else
            s += randexp(rng, T)
        end
        x .= Base.Fix1(quantile, d.dist).(x ./ s)
    end
    return x
end

# Moments

## Uniform

function mean(d::JointOrderStatistics{<:Uniform})
    return d.ranks .* scale(d.dist) ./ (d.n + 1) .+ minimum(d.dist)
end
function var(d::JointOrderStatistics{<:Uniform})
    n = d.n
    ranks = d.ranks
    return @. (scale(d.dist)^2 * ranks * (n + 1 - ranks)) / (n + 1)^2 / (n + 2)
end
function cov(d::JointOrderStatistics{<:Uniform})
    n = d.n
    ranks = d.ranks
    c = (scale(d.dist) / (n + 1))^2 / (n + 2)
    return broadcast(ranks, ranks') do rᵢ, rⱼ
        rmin, rmax = minmax(rᵢ, rⱼ)
        return rmin * (n + 1 - rmax) * c
    end
end

## Exponential

function mean(d::JointOrderStatistics{<:Exponential})
    # Arnold, eq 4.6.6
    θ = scale(d.dist)
    T = float(typeof(one(θ)))
    m = similar(d.ranks, T)
    ks = d.n .- d.ranks
    _harmonicnum!(m, ks)
    Hn = _harmonicnum_from(first(m), first(ks), d.n)
    @. m = θ * (Hn - m)
    return m
end
function var(d::JointOrderStatistics{<:Exponential})
    # Arnold, eq 4.6.7
    θ = scale(d.dist)
    T = float(typeof(oneunit(θ)^2))
    v = similar(d.ranks, T)
    ks = (d.n + 1) .- d.ranks
    _polygamma!(v, 1, ks)
    ϕn = _polygamma_from(1, first(v), first(ks), d.n + 1)
    @. v = θ^2 * (v - ϕn)
    return v
end
function cov(d::JointOrderStatistics{<:Exponential})
    # Arnold, eq 4.6.8
    v = var(d)
    S = broadcast(d.ranks, d.ranks', v, v') do rᵢ, rⱼ, vᵢ, vⱼ
        rᵢ < rⱼ ? vᵢ : vⱼ
    end
    return S
end

## Logistic

function mean(d::JointOrderStatistics{<:Logistic})
    # Arnold, eq 4.8.6
    T = typeof(oneunit(partype(d.dist)))
    m = H1 = similar(d.ranks, T)
    _harmonicnum!(H1, d.n .- d.ranks)
    if d.ranks == 1:(d.n)
        H2 = view(H1, reverse(eachindex(H1)))
    else
        H2 = similar(H1)
        _harmonicnum!(H2, d.ranks .- 1)
    end
    m .= scale(d.dist) .* (H2 .- H1) .+ mean(d.dist)
    return m
end
function var(d::JointOrderStatistics{<:Logistic})
    # Arnold, eq 4.8.7
    T = typeof(oneunit(partype(d.dist))^2)
    v = ϕ1 = similar(d.ranks, T)
    _polygamma!(ϕ1, 1, d.ranks)
    if d.ranks == 1:(d.n)
        ϕ2 = view(ϕ1, reverse(eachindex(ϕ1)))
    else
        ϕ2 = similar(ϕ1)
        _polygamma!(H2, 1, d.n + 1 - d.ranks)
    end
    v .= scale(d.dist)^2 .* (ϕ1 .+ ϕ2)
    return v
end
## Common utilities

# assume ns are sorted in increasing or decreasing order
function _harmonicnum!(Hns, ns::AbstractVector{<:Int})
    Hk = zero(eltype(Hns))
    k = 0
    iter = if last(ns) ≥ first(ns)
        zip(eachindex(Hns), ns)
    else
        Iterators.reverse(zip(eachindex(Hns), ns))
    end
    for (i, n) in iter
        Hns[i] = Hk = _harmonicnum_from(Hk, k, n)
        k = n
    end
    return Hns
end

# assume ns are sorted in increasing or decreasing order
function _polygamma!(ϕns, m::Int, ns::AbstractVector{<:Int})
    if last(ns) ≥ first(ns)
        i = lastindex(ϕns)
        k = last(ns)
        iter = Iterators.reverse(zip(eachindex(ϕns), ns))
    else
        i = firstindex(ϕns)
        k = first(ns)
        iter = zip(eachindex(ϕns), ns)
    end
    ϕns[i] = ϕk = polygamma(m, k)
    for (i, n) in Iterators.drop(iter, 1)
        ϕns[i] = ϕk = _polygamma_from(m, ϕk, k, n)
        k = n
    end
    return ϕns
end

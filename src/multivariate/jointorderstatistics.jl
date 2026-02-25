# Implementation based on chapters 2-4 of
# Arnold, Barry C., Narayanaswamy Balakrishnan, and Haikady Navada Nagaraja.
# A first course in order statistics. Society for Industrial and Applied Mathematics, 2008.

"""
    JointOrderStatistics <: MultivariateDistribution

The joint distribution of a subset of order statistics from a sample from a
univariate distribution.

    JointOrderStatistics(
        dist::UnivariateDistribution,
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
    D<:UnivariateDistribution,
    R<:Union{AbstractVector{Int},Tuple{Int,Vararg{Int}}},
    S<:ValueSupport,
} <: MultivariateDistribution{S}
    dist::D
    n::Int
    ranks::R
    function JointOrderStatistics(
        dist::UnivariateDistribution,
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
        return new{typeof(dist),typeof(ranks),value_support(typeof(dist))}(dist, n, ranks)
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

function logpdf(
    d::JointOrderStatistics{<:ContinuousUnivariateDistribution}, x::AbstractVector{<:Real}
)
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

# discrete case
# for y=unique(x), with known counts c, m=length(y), and parameters θ, the PMF is
# P(y,c|θ) = \sum_{d \in D(n, c)) P(d|n,p), where (taking y_0 = -Inf and y_{m+1} = Inf)
# - d_{2k}: the number of entries equal to y_k
# - d_{2k-1}: the number of entries in (y_k and y_{k-1})
# - p_{2k}: the probability of a draw equal to y_k (P(y_k|θ))
# - p_{2k-1}: the probability of a draw falling in (y_k, y_{k-1}) (P(y_k < x < y_{k-1}|θ))
# - D(n, c): the set of all weak 2m+1-compositions d of n (i.e. sum(d)=n) constrained by d_{2k} >= c_k
# - P(d|n,p)=Multinomial(d|n,p)
#
# The sum marginalizes over all possible count vectors d that satisfy the constraints implied by y and c.
# It's here computed efficiently as a product of Hankel matrices; since a Hankel matrix-vector product is
# equivalent to a discrete cross-correlation, we instead construct the defining sequences of the
# Hankel matrices and compute the cross-correlations in log-space.
function logpdf(
    d::JointOrderStatistics{<:DiscreteUnivariateDistribution}, x::AbstractVector{<:Real}
)
    (; n, ranks) = d
    udist = d.dist

    if length(ranks) == 1
        return logpdf(OrderStatistic(udist, n, first(ranks); check_args=false), first(x))
    end

    y, rank_ranges = _rle_ranks(x, ranks)

    if sum(length, rank_ranges) == n  # no gaps => all values are either observed or fixed by rank constraints
        # logpdf for Multinomial distribution over whole (potentially infinite) support
        lp = _log_hankel_base(n, Iterators.map(Base.Fix1(logpdf, udist), y), rank_ranges)
        issorted(x) && return lp
        return oftype(lp, -Inf)
    end

    log_tie_probs = logpdf.(Ref(udist), y)
    gap_lengths = _gap_lengths(n, rank_ranges)
    lp = _log_hankel_base(n, log_tie_probs, rank_ranges)

    # allocate workspaces
    max_gap_length = maximum(gap_lengths)
    max_total_gap_length = @views maximum(sum, zip(gap_lengths, gap_lengths[2:end]))
    T = eltype(lp)
    logh_work = similar(x, T, max_total_gap_length + 1)  # defining sequence for Hankel matrices of log-multinomial factors
    logv_work = similar(x, T, max_gap_length + 1)        # logsumexp of log-multinomial factors from left
    logc_work = similar(x, T, max_gap_length + 1)        # intermediate vector for log-cross-correlation
    init_state = (; logv_work, logc_work)

    _log_hankel_product_init!(init_state, udist, y, gap_lengths, log_tie_probs)
    (op!) = _make_log_hankel_product_op(logh_work, udist, y, rank_ranges, gap_lengths, log_tie_probs)
    final_state = foldl(op!, eachindex(y, log_tie_probs, rank_ranges); init=init_state)
    lp += first(final_state.logv_work)
    return lp
end

function _log_hankel_base(n, log_probs, rank_ranges)
    lp = sum(zip(log_probs, rank_ranges)) do (lp_i, range_i)
        num_ties_i = length(range_i)
        isone(num_ties_i) && return lp_i
        num_ties_i * lp_i - loggamma(oftype(lp_i, num_ties_i + 1))
    end
    return lp + loggamma(oftype(lp, n + 1))
end

function _log_hankel_product_init!(state, udist, y, gap_lengths, log_tie_probs)
    (; logv_work) = state
    T = eltype(logv_work)
    # initiate recurrence for left-flanking gap
    gap_length_left = gap_lengths[1]
    if gap_length_left == 0
        logv_work[begin] = 0
    else
        log_gap_prob = logsubexp(T(logcdf(udist, y[1])), log_tie_probs[1])
        logv = _view_first(logv_work, gap_length_left + 1)
        _log_gap_terms!(logv, log_gap_prob, gap_length_left)
    end
    return state
end

function _make_log_hankel_product_op(logh_work, udist, y, rank_ranges, gap_lengths, log_tie_probs)
    T = eltype(logh_work)
    ilast = lastindex(y)
    function log_hankel_product_op(state, i)
        (; logv_work, logc_work) = state
        gap_length_left = gap_lengths[i]
        gap_length_right = gap_lengths[i + 1]
        gap_length_total = gap_length_left + gap_length_right
        min_num_ties = length(rank_ranges[i])

        log_tie_prob = log_tie_probs[i]

        logv = _view_first(logv_work, gap_length_left + 1)
        logc = _view_first(logc_work, gap_length_right + 1)
        if gap_length_left == 0
            _log_tie_terms!(logc, log_tie_prob, min_num_ties, gap_length_right)
            logc .+= first(logv)
        else
            logh_ties = _view_first(logh_work, gap_length_total + 1)
            _log_tie_terms!(logh_ties, log_tie_prob, min_num_ties, gap_length_total)
            _log_xcorr_exp!(logc, logh_ties, logv)
        end

        if gap_length_right == 0
            logv_work, logc_work = logc_work, logv_work
            return (; logv_work, logc_work)
        end

        logh_gap = _view_first(logh_work, gap_length_right + 1)
        if i == ilast
            log_gap_prob = T(logccdf(udist, y[i]))
            # for right-flanking gap, logc is a row vector, and logh is a column vector, so
            # we only need an inner product (i.e. first term of a cross-correlation).
            logv = _view_first(logv_work, 1)
        else
            log_gap_prob = logsubexp(
                T(logdiffcdf(udist, y[i + 1], y[i])), log_tie_probs[i + 1]
            )
            logv = _view_first(logv_work, gap_length_right + 1)
        end
        _log_gap_terms!(logh_gap, log_gap_prob, gap_length_right)
        _log_xcorr_exp!(logv, logh_gap, logc)
        return (; logv_work, logc_work)
    end
    return log_hankel_product_op
end


_view_first(x, n) = @views x[begin:(begin - 1 + n)]

"""
    _rle_ranks(values, ranks) -> Tuple{Vector,Vector}

Return the run-length encoding of the order statistics at the specified ranks.

If we observe xj = xi for ranks rj > ri, then we know that all ranks between ri and rj
are also equal to xi, and they are included in the range even if they are not included in
`ranks`.

# Arguments
- `values`: Sorted vector of observed values
- `ranks`: Sorted vector of corresponding ranks (integer-valued)

# Returns
- `distinct_vals`: Vector of distinct values (sorted)
- `rank_ranges`: Vector of ranges of ranks for each distinct value
"""
function _rle_ranks(values, ranks)
    (val_last, rank_last), iter = Iterators.peel(zip(values, ranks))
    distinct_vals = eltype(values)[val_last]
    rank_ranges = UnitRange{eltype(ranks)}[]
    rank_first = rank_last
    for (val, rank) in iter
        if val != val_last
            push!(rank_ranges, rank_first:rank_last)
            push!(distinct_vals, val)
            rank_first = rank
        end
        val_last = val
        rank_last = rank
    end
    push!(rank_ranges, rank_first:rank_last)

    return distinct_vals, rank_ranges
end

"""
    _gap_lengths(n, rank_ranges) -> Vector{Int}

Compute the lengths of gaps between ranges of known ranks, including left- and right- tail gaps.
"""
function _gap_lengths(n::Integer, rank_ranges::Vector)
    gap_lengths = Vector{Int}(undef, length(rank_ranges) + 1)
    gap_lengths[1] = first(rank_ranges[1]) - 1
    for i in 2:length(rank_ranges)
        gap_lengths[i] = first(rank_ranges[i]) - last(rank_ranges[i - 1]) - 1
    end
    gap_lengths[end] = n - last(rank_ranges[end])
    return gap_lengths
end

"""
    _log_gap_terms!(logh, log_gap_prob, gap_size)

Compute the log-multinomial term for a gap between observed ranks (or tail gaps).

For a gap between observed ranks ``r_i < r_j`` (with ``x_i < x_j``) of size ``k_i = r_j - r_i + 1`` `=gap_size`,
where the probability of a draw falling in the gap is ``p_i = P(x_i < x < x_j)`` `=exp(log_gap_prob)`,
computes the logarithm of the multinomial terms
```math
h_{u+1} = p_i^{k_i - u} / (k_i - u)!
```
for ``u \\in [0, k_i]``.
"""
function _log_gap_terms!(logh, log_gap_prob, gap_size)
    T = eltype(logh)
    logh[end] = log_term = zero(T)
    accumulate!(@view(logh[end-1:-1:begin]), 1:gap_size; init=log_term) do log_term, num_in_gap
        return log_term + log_gap_prob - log(T(num_in_gap))
    end
    return logh
end

"""
    _log_tie_terms!(logh, log_tie_prob, min_num_ties, gap_size_total)

Compute the log-multinomial term for the ties with observed ranks from adjacent gaps.

Let ``x_{r:n}`` be the rank ``r`` order statistic of the sample ``x_1, ..., x_n``.
For a block of known ranks ``r_{i}...r_{i+c_i-1}``
(with ``x_{r_{i-1}:n} < x_{r_i:n} = ... = x_{r_{i+c_i-1}:n} < x_{r_{i+c_i}:n}``)
of size ``c_i`` `=min_num_ties`, flanked by gaps with sizes ``k_{i-1}`` and ``k_i`` and
total gap size ``g_i = k_{i-1} + k_i`` `=gap_size_total`,
computes the logarithm of the multinomial terms ``h`` where
```math
h_{u+1} (f(x_i)^{c_i} / c_i!) = f(x_i)^{c_i + u} / (c_i + u)!,
```
and ``f(x_i)`` `=exp(log_tie_prob)``, for ``u \\in [0, g_i]``.
"""
function _log_tie_terms!(logh, log_tie_prob, min_num_ties, gap_size_total)
    T = eltype(logh)
    logh[begin] = log_term = zero(T)
    accumulate!(@view(logh[begin+1:end]), 1:gap_size_total; init=log_term) do log_term, num_ties_gap
        num_ties_total = num_ties_gap + min_num_ties
        return log_term + log_tie_prob - log(T(num_ties_total))
    end
    return logh
end


"""
    _log_xcorr_exp!(log_c, log_a, log_b)

Compute in-place the logarithm of the cross-correlation of the exponential of `log_a` and `log_b`.

```math
\\log(c_j) = \\log(\\sum_i \\exp(\\log(a_i) + \\log(b_{i+j-1})))
```
with implicit `-Inf`-padding of `log_a` and `log_b` to the right as needed.

Only the requested entries of `log_c` are computed.

Note: this is equivalent to but more numerically stable than passing 0-indexed offset arrays for
`a` and `b` to `DSP.xcorr`, truncating the result `c` to `c[0:length(log_c)-1]`, and taking the
logarithm of the result.

# Arguments
- `log_c`: Vector to store the result
- `log_a`: Vector of logarithms of the first factor
- `log_b`: Vector of logarithms of the second factor
"""
function _log_xcorr_exp!(log_c, log_a, log_b)
    n_r = length(log_b)
    idx_last = lastindex(log_a)
    map!(log_c, first(eachindex(log_a), length(log_c))) do j_idx
        terms = Iterators.map(+, log_b, @views log_a[j_idx:min(j_idx + n_r - 1, idx_last)])
        return logsumexp(terms)
    end
    return log_c
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

        u = eltype(x) <: Integer ? similar(x, float(eltype(x))) : x

        T = typeof(one(eltype(u)))
        s = zero(eltype(u))
        i = 0
        for (m, j) in zip(eachindex(u), d.ranks)
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
            u[m] = s
        end
        j = n + 1
        k = j - i
        if k > 1
            s += T(rand(rng, GammaMTSampler(Gamma{T}(T(k), T(1)))))
        else
            s += randexp(rng, T)
        end
        x .= Base.Fix1(quantile, d.dist).(u ./ s)
    end
    return x
end

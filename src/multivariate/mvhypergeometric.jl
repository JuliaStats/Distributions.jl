"""
The [Multivariate hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution#Multivariate_hypergeometric_distribution)
generalizes the *hypergeometric distribution*. Consider ``n`` draws from a finite population containing ``k`` types of elements. Suppose that the population has size ``M`` and there are ``m_i`` elements of type ``i`` for ``i = 1, .., k`` with ``m_1+...m_k = M``. Let ``X = (X_1, ..., X_k)`` where ``X_i`` represents the number of elements of type ``i`` drawn, then the distribution of ``X`` is a multivariate hypergeometric distribution. Each sample of a multivariate hypergeometric distribution is a ``k``-dimensional integer vector that sums to ``n`` and satisfies ``0 \\le X_i \\le m_i``. 


The probability mass function is given by

```math
f(x; m, n) = {{{m_1 \\choose x_1}{m_2 \\choose x_2}\\cdots {m_k \\choose x_k}}\\over {N \\choose n}}, 
\\quad x_1 + \\cdots + x_k = n, \\quad x_i \\le m_i
```

```julia
MvHypergeometric(m, n)   # Multivariate hypergeometric distribution for a population with
                         # m = (m_1, ..., m_k) elements of type 1 to k and n draws
```
"""
struct MvHypergeometric <: DiscreteMultivariateDistribution
    m::Vector{Int}  # number of elements of each type
    n::Int                  # number of draws
    function MvHypergeometric(m::Vector{Int}, n::Int; check_args::Bool=true)
        @check_args(
            MvHypergeometric,
            (m, all(x -> x >= 0, m)),
            zero(n) <= n <= sum(m),
        )
        new(m, n)
    end
end


# Parameters

ncategories(d::MvHypergeometric) = length(d.m)
length(d::MvHypergeometric) = ncategories(d)
ntrials(d::MvHypergeometric) = d.n

params(d::MvHypergeometric) = (d.m, d.n)
partype(::MvHypergeometric) = Int

# Statistics

mean(d::MvHypergeometric) = d.n .* d.m ./ sum(d.m)

function var(d::MvHypergeometric)
    m = d.m
    n = ntrials(d)
    M = sum(m)
    f = n * (M - n) / (M - 1)
    v = let f = f
        map(mi -> f * (mi / M) * ((M - mi) / M), m)
    end
    v
end

function cov(d::MvHypergeometric)
    m = d.m
    n = ntrials(d)
    M = sum(m)
    p = m / M
    f = n * (M - n) / (M - 1)

    C = -f * (p * p')
    C[diagind(C)] .= f .* p .* (1 .- p)

    C
end


# Evaluation
function insupport(d::MvHypergeometric, x::AbstractVector{<:Real})
    k = length(d)
    m = d.m
    length(x) == k || return false
    s = 0.0
    for i = 1:k
        @inbounds xi = x[i]
        if !(isinteger(xi) && xi >= 0 && xi <= m[i])
            return false
        end
        s += xi
    end
    return s == ntrials(d)  # integer computation would not yield truncation errors
end

function _logpdf(d::MvHypergeometric, x::AbstractVector{<:Real})
    m = d.m
    M = sum(m)
    n = ntrials(d)
    insupport(d, x) || return -Inf
    s = -logabsbinomial(M, n)[1]
    for i = 1:length(m)
        @inbounds xi = x[i]
        @inbounds m_i = m[i]
        s += logabsbinomial(m_i, xi)[1]
    end
    return s
end

# Sampling is performed by sequentially sampling each entry from the
# hypergeometric distribution
_rand!(rng::AbstractRNG, d::MvHypergeometric, x::AbstractVector{<:Real}) =
    mvhypergeom_rand!(rng, d.m, ntrials(d), x)





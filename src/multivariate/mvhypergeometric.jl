"""
The [Multivariate hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution#Multivariate_hypergeometric_distribution)
generalizes the *hypergeometric distribution*. Consider ``n`` draws from a finite population containing `k` types of elements. Suppose that the population has size `M` and there are ``m_i`` elements of type ``i`` for ``i = 1, .., k`` with ``m_1+...m_k = M``. Let ``X = (X_1, ..., X_k)`` where ``X_i`` represents the number of elements of type ``i`` drawn, then the distribution of ``X`` is a multivariate hypergeometric distribution. Each sample of a multivariate hypergeometric distribution is a ``k``-dimensional integer vector that sums to ``n``. 


The probability mass function is given by

```math
f(x; m, n) = {{{m_1 \\choose x_1}{m_2 \\choose x_2}\\cdots {m_k \\choose x_k}}\\over {N \\choose n}}, 
\\quad x_1 + \\cdots + x_k = n, \\quad x_i \\le m_i
```

```julia
MvHypergeometric(m, n)   # Multivariate hypergeometric distribution for a population with
                         # m = (m_1, ..., m_k) elements of type 1 to k and n draws
                         
params(d)       # Get the parameters, i.e. (m, n)
```
"""
struct MvHypergeometric <: DiscreteMultivariateDistribution
    m::AbstractVector{Int}  # number of elements of each type
    n::Int                  # number of draws
    function MvHypergeometric(m::Real, n::Real; check_args::Bool=true)
        @check_args(
            MvHypergeometric,
            (m, m >= zero.(n)),
            zero(n) <= n <= sum(m),
        )
        new(m, n)
    end
end


# Parameters

ncategories(d::MvHypergeometric) = length(d.m)
length(d::MvHypergeometric) = ncategories(d)
nelements(d::MvHypergeometric) = d.m
ntrials(d::MvHypergeometric) = d.n

params(d::MvHypergeometric) = (d.m, d.n)

# Statistics

mean(d::MvHypergeometric) = d.n .* d.m ./ sum(d.m)

function var(d::MvHypergeometric{T}) where T<:Real
    m = nelements(d)
    k = length(m)
    n = ntrials(d)
    M = sum(m)
    p = m / M
    f = n * (M - n) / (M-1)

    v = Vector{T}(undef, k)
    for i = 1:k
        @inbounds p_i = p[i]
        v[i] = f * p_i * (1 - p_i)
    end
    v
end

function cov(d::MvHypergeometric{T}) where T<:Real
    m = nelements(d)
    k = length(m)
    n = ntrials(d)
    M = sum(m)
    p = m / M
    f = n * (M - n) / (M-1)

    C = Matrix{T}(undef, k, k)
    for j = 1:k
        pj = p[j]
        for i = 1:j-1
            @inbounds C[i,j] = - f * p[i] * pj
        end

        @inbounds C[j,j] = f * pj * (1-pj)
    end

    for j = 1:k-1
        for i = j+1:k
            @inbounds C[i,j] = C[j,i]
        end
    end
    C
end


# Evaluation
function insupport(d::MvHypergeometric, x::AbstractVector{T}) where T<:Real
    k = length(d)
    m = nelements(d)
    length(x) == k || return false
    s = 0.0
    for i = 1:k
        @inbounds xi = x[i]
        if !(isinteger(xi) && xi >= 0 && x <= m[i])
            return false
        end
        s += xi
    end
    return s == ntrials(d)  # integer computation would not yield truncation errors
end

function _logpdf(d::MvHypergeometric, x::AbstractVector{T}) where T<:Real
    m = nelements(d)
    M = sum(m)
    n = ntrials(d)
    insupport(d,x) || return -Inf
    s = -logabsbinomial(M, n)[1]
    for i = 1:length(m)
        @inbounds xi = x[i]
        @inbounds m_i = m[i]
        s += logabsbinomial(m_i, xi)[1]
    end
    return s
end

# Testing it out
n = 5
m = [5, 3, 2]

d = MvHypergeometric(m, n)
x = [2, 2, 1]

println("Probability of $x: ", exp(_logpdf(d, x)))


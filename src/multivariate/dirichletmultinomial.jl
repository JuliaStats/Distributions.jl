"""
    DirichletMultinomial

The [Dirichlet-multinomial distribution](https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution)
is the distribution of a draw from a multinomial distribution where each sample has a 
slightly different probability vector, drawn from a common Dirichlet distribution.

This contrasts with the multinomial distribution, which assumes that all observations arise
from a single fixed probability vector. This enables the Dirichlet-multinomial distribution to
accommodate more variable (a.k.a, over-dispersed) count data than the multinomial distribution.

The probability mass function is given by

```math
f(x; \\alpha) = \\frac{n! \\Gamma(\\alpha_0)}
{\\Gamma(n+\\alpha_0)}\\prod_{k=1}^K\\frac{\\Gamma(x_{k}+\\alpha_{k})}
{x_{k}! \\Gamma(\\alpha_{k})}
```
where
- ``n = \\sum_k x_k``
- ``\\alpha_0 = \\sum_k \\alpha_k``

```julia
# Let α be a vector
DirichletMultinomial(n, α) # Dirichlet-multinomial distribution for n trials with parameter
vector α.

# Let k be a positive integer
DirichletMultinomial(n, k) # Dirichlet-multinomial distribution with n trials and parameter
vector of length k of ones.
```
"""
struct DirichletMultinomial{T <: Real} <: DiscreteMultivariateDistribution
    n::Int
    α::Vector{T}
    α0::T

    function DirichletMultinomial{T}(n::Integer, α::Vector{T}) where T
        α0 = sum(abs, α)
        sum(α) == α0 || throw(ArgumentError("alpha must be a positive vector."))
        n > 0 || throw(ArgumentError("n must be a positive integer."))
        new{T}(Int(n), α, α0)
    end
end
DirichletMultinomial(n::Integer, α::Vector{T}) where {T <: Real} = DirichletMultinomial{T}(n, α)
DirichletMultinomial(n::Integer, α::Vector{T}) where {T <: Integer} = DirichletMultinomial(n, float(α))
DirichletMultinomial(n::Integer, k::Integer) = DirichletMultinomial(n, ones(k))

Base.show(io::IO, d::DirichletMultinomial) = show(io, d, (:n, :α,))


# Parameters
ncategories(d::DirichletMultinomial) = length(d.α)
length(d::DirichletMultinomial) = ncategories(d)
ntrials(d::DirichletMultinomial) = d.n
params(d::DirichletMultinomial) = (d.n, d.α)
@inline partype(d::DirichletMultinomial{T}) where {T} = T

# Statistics
mean(d::DirichletMultinomial) = d.α .* (d.n / d.α0)
function var(d::DirichletMultinomial{T}) where T <: Real
    v = fill(d.n * (d.n + d.α0) / (1 + d.α0), length(d))
    p = d.α / d.α0
    for i in eachindex(v)
        @inbounds v[i] *= p[i] * (1 - p[i])
    end
    v
end
function cov(d::DirichletMultinomial{<:Real})
    v = var(d)
    c = d.α * d.α'
    lmul!(-d.n * (d.n + d.α0) / (d.α0^2 * (1 + d.α0)), c)
    for (i, vi) in zip(diagind(c), v)
        @inbounds c[i] = vi
    end
    c
end


# Evaluation
function insupport(d::DirichletMultinomial, x::AbstractVector{T}) where T<:Real
    k = length(d)
    length(x) == k || return false
    for xi in x
        (isinteger(xi) && xi >= 0) || return false
    end
    return sum(x) == ntrials(d)
end
function _logpdf(d::DirichletMultinomial{S}, x::AbstractVector{T}) where {T<:Real, S<:Real}
    c = loggamma(S(d.n + 1)) + loggamma(d.α0) - loggamma(d.n + d.α0)
    for j in eachindex(x)
        @inbounds xj, αj = x[j], d.α[j]
        c += loggamma(xj + αj) - loggamma(xj + 1) - loggamma(αj)
    end
    c
end


# Sampling
_rand!(rng::AbstractRNG, d::DirichletMultinomial, x::AbstractVector{<:Real}) =
    multinom_rand!(rng, ntrials(d), rand(rng, Dirichlet(d.α)), x)

# Fit Model
# Using https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2945396/pdf/nihms205488.pdf
struct DirichletMultinomialStats <: SufficientStats
    n::Int
    s::Matrix{Float64}  # s_{jk} = ∑_i x_{ij} ≥ (k - 1),  k = 1,...,(n - 1)
    tw::Float64
    DirichletMultinomialStats(n::Int, s::Matrix{Float64}, tw::Real) = new(n, s, Float64(tw))
end
function suffstats(::Type{<:DirichletMultinomial}, x::Matrix{T}) where T<:Real
    ns = sum(x, dims=1)  # get ntrials for each observation
    n = ns[1]       # use ntrails from first ob., then check all equal
    all(ns .== n) || error("Each sample in X should sum to the same value.")
    d, m = size(x)
    s = zeros(d, n)
    @inbounds for k in 1:n, i in 1:m, j in 1:d
        if x[j, i] >= k
            s[j, k] += 1.0
        end
    end
    DirichletMultinomialStats(n, s, m)
end
function suffstats(::Type{<:DirichletMultinomial}, x::Matrix{T}, w::Array{Float64}) where T<:Real
    length(w) == size(x, 2) || throw(DimensionMismatch("Inconsistent argument dimensions."))
    ns = sum(x, dims=1)
    n = ns[1]
    all(ns .== n) || error("Each sample in X should sum to the same value.")
    d, m = size(x)
    s = zeros(d, n)
    @inbounds for k in 1:n, i in 1:m, j in 1:d
        if x[j, i] >= k
            s[j, k] += w[i]
        end
    end
    DirichletMultinomialStats(n, s, sum(w))
end
function fit_mle(::Type{<:DirichletMultinomial}, ss::DirichletMultinomialStats;
                 tol::Float64 = 1e-8, maxiter::Int = 1000)
    k = size(ss.s, 2)
    α = ones(size(ss.s, 1))
    rng = 0.0:(k - 1)
    @inbounds for iter in 1:maxiter
        α_old = copy(α)
        αsum = sum(α)
        denom = ss.tw * sum(inv, αsum .+ rng)
        for j in eachindex(α)
            αj = α[j]
            num = αj * sum(vec(ss.s[j, :]) ./ (αj .+ rng))  # vec needed for 0.4
            α[j] = num / denom
        end
        maximum(abs, α_old - α) < tol && break
    end
    DirichletMultinomial(ss.n, α)
end

immutable DirichletMultinomial{T <: Real} <: DiscreteMultivariateDistribution
    n::Int
    α::Vector{T}
    α0::T

    function DirichletMultinomial(n::Integer, α::Vector{T})
        α0 = sumabs(α)
        sum(α) == α0 || throw(ArgumentError("DirichletMultinomial: alpha must be a positive vector."))
        n > 0 || throw(ArgumentError("DirichletMultinomial: n must be a positive integer."))
        new(Int(n), α, α0)
    end
end
DirichletMultinomial{T <: Real}(n::Integer, α::Vector{T}) = DirichletMultinomial{T}(n, α)
DirichletMultinomial{T <: Integer}(n::Integer, α::Vector{T}) = DirichletMultinomial(n, float(α))
DirichletMultinomial(n::Integer, k::Integer) = DirichletMultinomial(n, ones(k))

Base.show(io::IO, d::DirichletMultinomial) = show(io, d, (:n, :α,))


# Parameters
ncategories(d::DirichletMultinomial) = length(d.α)
length(d::DirichletMultinomial) = ncategories(d)
ntrials(d::DirichletMultinomial) = d.n
params(d::DirichletMultinomial) = (d.n, d.α)
@inline partype{T<:Real}(d::DirichletMultinomial{T}) = T


# Statistics
mean(d::DirichletMultinomial) = d.α .* (d.n / d.α0)
function var{T <: Real}(d::DirichletMultinomial{T})
    v = fill(d.n * (d.n + d.α0) / (one(T) + d.α0), length(d))
    p = d.α / d.α0
    for i in eachindex(v)
        @inbounds v[i] *= p[i] * (one(T) - p[i])
    end
    v
end
function cov{T <: Real}(d::DirichletMultinomial{T})
    v = var(d)
    c = d.α * d.α'
    multiply!(c, -(d.n / d.α0 ^ 2) * ((d.n + d.α0) / (one(T) + d.α0)))
    for i in 1:length(d)
        @inbounds c[i, i] = v[i]
    end
    c
end


# Evaluation
function insupport{T<:Real}(d::DirichletMultinomial, x::AbstractVector{T})
    k = length(d)
    length(x) == k || return false
    for xi in x
        (isinteger(xi) && xi >= 0) || return false
    end
    return sum(x) == ntrials(d)
end
function _logpdf{T<:Real}(d::DirichletMultinomial, x::AbstractVector{T})
    c = log(factorial(d.n)) + lgamma(d.α0) - lgamma(d.n + d.α0)
    for j in eachindex(x)
        xj, αj = x[j], d.α[j]
        c += lgamma(xj + αj) - log(factorial(xj)) - lgamma(αj)
    end
    c
end


# Sampling
function _rand!{T<:Real}(d::DirichletMultinomial, x::AbstractVector{T})
    multinom_rand!(ntrials(d), rand(Dirichlet(d.α)), x)
end


# Fit Model
# Using https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2945396/pdf/nihms205488.pdf
immutable DirichletMultinomialStats <: SufficientStats
    n::Int
    s::Matrix{Float64}  # s_{jk} = ∑_i x_{ij} ≥ (k - 1),  k = 1,...,(n - 1)
    tw::Float64
    DirichletMultinomialStats(n::Int, s::Matrix{Float64}, tw::Real) = new(n, s, Float64(tw))
end
function suffstats{T<:Real}(::Type{DirichletMultinomial}, x::Matrix{T})
    n = sum(x[:, 1])
    all(sum(x, 1) .== n) || error("Each sample in X should sum to the same value.")
    # s = [float(sum(x[i, :] .>= k)) for i in 1:size(x, 1), k in 1:n]
    s = zeros(size(x, 1), n)
    for k in 1:n
        for i in 1:size(x, 2)
            for j in 1:size(x, 1)
                s[j, k] += Float64(x[j, i] >= k)
            end
        end
    end
    DirichletMultinomialStats(n, s, size(x, 2))
end
function fit_mle(::Type{DirichletMultinomial}, ss::DirichletMultinomialStats;
        tol::Float64 = 1e-8, maxiter::Int = 1000)
    k = size(ss.s, 2)
    α = ones(size(ss.s, 1))
    rng = 0.0:(k - 1)
    tolerance = Inf
    for iter in 1:maxiter
        α_old = copy(α)
        αsum = sum(α)
        denom = ss.tw * sum(1 ./ (αsum + rng))
        for j in eachindex(α)
            αj = α[j]
            num = αj * sum(ss.s[j, :] ./ (αj + rng))
            α[j] = num / denom
        end
        tolerance = maxabs(α_old - α)
        if tolerance < tol
            println("done in $iter iterations")
            break
        end
    end
    println("tol: $tolerance")
    DirichletMultinomial(ss.n, α)
end

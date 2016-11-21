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
# TODO: use https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2945396/pdf/nihms205488.pdf
# What gets returned if there are different number of trials in observations?
#   - Not an option for Multinomial
# immutable DirichletMultinomialStats <: SufficientStats
#     r::Vector{Int}
#     s::Matrix{Int}
# end

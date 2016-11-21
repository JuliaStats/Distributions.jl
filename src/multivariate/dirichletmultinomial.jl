immutable DirichletMultinomial{T <: Real} <: ContinuousMultivariateDistribution
    n::Int
    α::Vector{T}
    α0::T

    function DirichletMultinomial(n::Integer, α::Vector{T})
        α0 = sumabs(α)
        sum(α) == α0 || throw(ArgumentError("DirichletMultinomial: alpha must be a positive vector."))
        n > 0 || throw(ArgumentError("DirichletMultinomial: n must be a positive integer."))
        new(round(Int, n), α, α0)
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
function cov(d::DirichletMultinomial)

end

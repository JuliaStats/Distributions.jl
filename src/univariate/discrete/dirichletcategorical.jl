"""
    DirichletCategorical(α,n)

A *Dirichlet-categorical distribution* is the compound distribution of the [`Categorical`](@ref) distribution where the probability of each category is distributed according to the [`Dirichlet`](@ref). It has two parameters: the concentration parameters `α` and the optional observation counts `n`.

```math
p(x = i \\vert n, \\alpha) = \\int\\limits_p p(x \\vert p) p(p \\vert n, \\alpha) dp = \\frac{n_i + \\alpha_1}{\\sum\\limits_k n_k + α_k}
```

```julia
DirichletCategorical(α, n)      # DirichcletCategorical distribution with concentration parameters α and observation counts n.

params(d)       # Get the parameters, i.e. (α, n)
update!(d, o)   # Update the counts d.n with observation(s) o.
```

External links:

* [Categorical distribution conjugate prior on Wikipedia](https://en.wikipedia.org/wiki/Categorical_distribution#Bayesian_inference_using_conjugate_prior)
"""
struct DirichletCategorical{T <: Real, U <: Integer} <: DiscreteUnivariateDistribution
    α::Vector{T}
    α0::T
    n::Vector{U}

    function DirichletCategorical(α::Vector{T}, n::Vector{U}) where {T, U}
        α0 = sum(abs, α)
        N = sum(abs, n)
        sum(α) == α0 || throw(ArgumentError("alpha must be a positive vector."))
        sum(n) == N || throw(ArgumentError("n must be a positive vector."))
        length(n) == length(α) || throw(ArgumentError("n and α must be of same length"))
        new{T,U}(α, α0, n)
    end
end

DirichletCategorical(α::Vector{<: Integer}, n::Vector{<: Integer})= DirichletCategorical(float(α), n)
DirichletCategorical(k::Integer) = DirichletCategorical(ones(k), zeros(Int, k))
DirichletCategorical(α::Vector{<: Real}) = DirichletCategorical(α, zeros(Int, length(α)))

Base.show(io::IO, d::DirichletCategorical) = show(io, d, (:α, :n))

ncategories(d::DirichletCategorical) = length(d.α)
length(d::DirichletCategorical) = ncategories(d)
params(d::DirichletCategorical) = (d.α, d.n)
@inline partype(d::DirichletCategorical{T}) where {T} = T

function rand(rng::AbstractRNG, d::DirichletCategorical)
    quantile(d, rand(rng))
end

function pdf(d::DirichletCategorical, x::Int)
    if insupport(d, x)
        @inbounds (d.α[x] + d.n[x]) / (d.α0 + sum(d.n))
    else
        zero(x)
    end
end

function logpdf(d::DirichletCategorical, x::Int)
    if insupport(d, x)
        @inbounds log(d.α[x] + d.n[x]) - log(d.α0 + sum(d.n))
    else
        -Inf
    end
end

function cdf(d::DirichletCategorical, x::Int)
    if x < minimum(d)
        zero(x)
    elseif x > maximum(d)
        one(x)
    else
        @inbounds sum((d.α[1:x] + d.n[1:x]) / (d.α0 + sum(d.n)))
    end
end

function quantile(d::DirichletCategorical, q::Real)
    0 <= q <= 1 || throw(DomainError(q, "q ∈ [0,1]"))
    p = probs(d)
    cp = p[1]
    i = 1
    while cp < q
        i += 1
        @inbounds cp += p[i]
    end
    i
end

probs(d::DirichletCategorical) = (d.n + d.α) .* inv(d.α0 + sum(d.n))

function minimum(d::DirichletCategorical)
    return 1
end

function maximum(d::DirichletCategorical)
    return length(d.α)
end

function insupport(d::DirichletCategorical, x::Real)
    return x >= minimum(d) && x <= maximum(d)
end

function mode(d::DirichletCategorical)
    m, i = findmax(dirichlet_mode(d.α + d.n, d.α0 + sum(d.n)))
    i
end

function update!(d::DirichletCategorical, obs::Int)
    insupport(d, obs) || throw(ArgumentError("out of bounds observation"))

    @inbounds d.n[obs] += 1
end

function update!(d::DirichletCategorical, obs::Vector{Int})
    all(map(x -> insupport(d, x), obs)) || throw(ArgumentError("out of bounds observation"))

    for o in obs
        @inbounds d.n[o] += 1
    end
end

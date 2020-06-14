"""

Soliton(K::Integer, M::Integer, δ::Real, atol::Real=0) <: Distribution{Univariate, Discrete}

The Robust Soliton distribution of length `K`, mode `M` (i.e., the
location of the robust component spike), peeling process failure
probability `δ`, and minimum non-zero probability mass `atol`. More
specifically, degrees `i` for which `pdf(Ω, i)<atol` are set to
`0`. Letting `atol=0` yields the regular robust Soliton distribution.

```julia
Soliton(K, M, δ)        # Robust Soliton distribution (with atol=0)
Soliton(K, M, δ, atol)  # Robust Soliton distribution with minimum non-zero probability mass atol

params(Ω)               # Get the parameters ,i.e., (K, M, δ, atol)
degrees(Ω)              # Return a vector composed of the degrees with non-zero probability mass
pdf(Ω, i)               # Evaluate the pdf at i
cdf(Ω, i)               # Evaluate the pdf at i
rand(Ω)                 # Sample from Ω
rand(Ω, n)              # Draw n samples from Ω
```

External links:

* [Soliton distribution on Wikipedia](https://en.wikipedia.org/wiki/Soliton_distribution)

"""
struct Soliton <: DiscreteUnivariateDistribution
    K::Int64 # Number of input symbols
    M::Int64 # Location of the robust component spike
    δ::Float64 # Peeling process failure probability
    atol::Float64 # Minimum non-zero probability assigned to a degree
    degrees::Vector{Int} # Degrees with non-zero probability
    CDF::Vector{Float64} # CDF evaluated at each element in degrees
    function Soliton(K::Integer, M::Integer, δ::Real, atol::Real=0)
        0 < K || throw(ArgumentError("Expected 0 < K, but got $K."))
        0 < δ < 1 || throw(ArgumentError("Expected 0 < δ < 1, but got $δ."))
        0 < M <= K || throw(ArgumentError("Expected 0 < M <= K, but got $M."))
        0 <= atol < 1 || throw(DomainError(atol, "Expected 0 <= atol < 1."))
        PDF = [τ(K, M, δ, i)+ρ(K, i) for i in 1:K]
        PDF ./= sum(PDF)
        degrees = [i for i in 1:K if PDF[i] > atol]
        CDF = cumsum([PDF[i] for i in degrees])
        CDF ./= CDF[end]
        new(K, M, δ, atol, degrees, CDF)
    end
end

Base.show(io::IO, Ω::Soliton) = print(io, "Soliton(K=$(Ω.K), M=$(Ω.M), δ=$(Ω.δ), atol=$(Ω.atol))")

"""
    degrees(Ω)

Return a vector composed of the degrees with non-zero probability.

"""
degrees(Ω::Soliton) = copy(Ω.degrees)

"""Robust component of the Soliton distribution."""
function τ(K::Integer, M::Integer, δ::Real, i::Integer)::Float64
    i <= K || throw(ArgumentError("Expected i <=K, but got $i."))
    R = K / M
    if i < M
        return 1 / (i * M)
    elseif i == M
        return log(R / δ) / M
    else # i <= K
        return 0.0
    end
end

"""Ideal component of the Soliton distribution."""
function ρ(K::Integer, i::Integer)::Float64
    i <= K || throw(ArgumentError("Expected i <=K, but got $i."))
    if i == 1
        return 1 / K
    else # i <= K
        return 1 / (i * (i - 1))
    end
end

StatsBase.params(Ω::Soliton) = (Ω.K, Ω.M, Ω.δ, Ω.atol)

function Distributions.pdf(Ω::Soliton, i::Integer)::Float64
    j = searchsortedfirst(Ω.degrees, i)
    if j > length(Ω.degrees) || Ω.degrees[j] != i return 0.0 end
    rv = Ω.CDF[j]
    if j > 1 rv -= Ω.CDF[j-1] end
    return rv
end

function Distributions.cdf(Ω::Soliton, i::Integer)
    i < Ω.degrees[1] && return 0.0
    i > Ω.degrees[end] && return 1.0
    j = searchsortedfirst(Ω.degrees, i)
    Ω.degrees[j] == i && return Ω.CDF[j]
    return Ω.CDF[j-1]
end

Statistics.mean(Ω::Soliton) = sum(i -> i*pdf(Ω, i), Ω.degrees)

function Statistics.var(Ω::Soliton)
    μ = mean(Ω)
    rv = 0.0
    for d in Ω.degrees
        rv += pdf(Ω, d)*(d-μ)^2
    end
    return rv
end

function Statistics.quantile(Ω::Soliton, v::Real)::Int
    0 <= v <= 1 || throw(ArgumentError("Expected 0 <= v <= 1, but got $v."))
    j = searchsortedfirst(Ω.CDF, v)
    j = min(length(Ω.degrees), j)
    return Ω.degrees[j]
end

Base.minimum(Ω::Soliton) = 1
Base.maximum(Ω::Soliton) = Ω.K

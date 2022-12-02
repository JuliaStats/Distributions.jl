# Corresponds to https://en.wikipedia.org/wiki/Logarithmic_distribution
struct Logarithmic{T<:Real} <: DiscreteUnivariateDistribution
    p::T
    function Logarithmic(p::T) where {T <: Real}
        new{T}(p)
    end
end
function logpdf(d::Logarithmic, x::Real)
    insupport(d, x) ? x*log(d.p) - log(x) - log(-log(1-d.p)) : log(zero(d.p))
end
function rand(rng::AbstractRNG, d::Logarithmic)
    # Sample a Log(p) distribution with the algorithm "LK" of Kemp (1981).
    u = rand(rng)
    if u > d.p 
        return 1
    end
    q = 1 - (1-d.p)^rand(rng)
    if u < q*q
        return floor(1+log(u)/log(q))
    end
    if u < q
        return 1
    end
    return 2
end
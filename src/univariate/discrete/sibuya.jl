# See https://rdrr.io/rforge/copula/man/Sibuya.html
struct Sibuya{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    p::T
    function Sibuya(p::T) where {T <: Real}
        new{T}(p)
    end
end
function Distributions.rand(rng::Distributions.AbstractRNG, d::Sibuya{T}) where {T <: Real}
    u = rand(rng, T)
    if u <= d.p
        return T(1)
    end
    xMax = 1/eps(T)
    Ginv = ((1-u)*SpecialFunctions.gamma(1-d.p))^(-1/d.p)
    fGinv = floor(Ginv)
    if Ginv > xMax 
        return fGinv
    end
    if 1-u < 1/(fGinv*SpecialFunctions.beta(fGinv,1-d.p))
        return ceil(Ginv)
    end
    return fGinv
end

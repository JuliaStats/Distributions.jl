#### Deprecate on 0.6 (to be removed on 0.7)

@Base.deprecate expected_logdet meanlogdet

function probs(d::DiscreteUnivariateDistribution)
    Base.depwarn("probs(d::$(typeof(d))) is deprecated. Please use pdf(d) instead.", :probs)
    return probs(d)
end

function Binomial(n::Real, p::Real)
    Base.depwarn("Binomial(n::Real, p) is deprecated. Please use Binomial(n::Integer, p) instead.", :Binomial)
    Binomial(Int(n), p)
end

function Binomial(n::Real)
    Base.depwarn("Binomial(n::Real) is deprecated. Please use Binomial(n::Integer) instead.", :Binomial)
    Binomial(Int(n))
end

function BetaBinomial(n::Real, α::Real, β::Real)
    Base.depwarn("BetaBinomial(n::Real, α, β) is deprecated. Please use BetaBinomial(n::Integer, α, β) instead.", :BetaBinomial)
    BetaBinomial(Int(n), α, β)
end



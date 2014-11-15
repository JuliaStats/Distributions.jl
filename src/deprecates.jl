
#### Deprecate on 0.5 (to be removed on 0.6)

function dim(d::MultivariateDistribution)
    Base.depwarn("dim(d::MultivariateDistribution) is deprecated. Please use length(d).", :dim)
    return length(d)
end

function binaryentropy(d::UnivariateDistribution) 
    Base.depwarn("binaryentropy is deprecated. Please use entropy(d, 2).", :binaryentropy)
    return entropy(d) / log(2)
end

@Base.deprecate logpmf logpdf
@Base.deprecate logpmf! logpmf!
@Base.deprecate pmf pdf


#### Deprecate on 0.6 (to be removed on 0.7)

@Base.deprecate expected_logdet meanlogdet

function probs(d::DiscreteUnivariateDistribution)
    Base.depwarn("probs(d::$(typeof(d))) is deprecated. Please use pdf(d) instead.", :probs)
    return probs(d)
end


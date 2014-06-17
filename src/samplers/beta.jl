sampler(d::Beta) = BetaGamSampler(d::Beta)
rand(d::Beta) = rand(sampler(d))
rand!(d::Beta,a::Array) = rand!(sampler(d),a)

immutable BetaGamSampler{Ga,Gb} <: Sampler{Univariate,Continuous}
    alpha::Ga
    beta::Gb
end

BetaGamSampler(d::Beta) = BetaGamSampler(sampler(Gamma(d.alpha)),sampler(Gamma(d.beta)))

function rand(s::BetaGamSampler)
    u = rand(s.alpha)
    v = rand(s.beta)
    u/(u+v)
end

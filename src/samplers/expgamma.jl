# These are used to bypass subnormals when sampling from

# Inverse Power sampler
# uses the x*u^(1/a) trick from Marsaglia and Tsang (2000) for when shape < 1
struct ExpGammaIPSampler{S<:Sampleable{Univariate,Continuous},T<:Real} <: Sampleable{Univariate,Continuous}
    s::S #sampler for Gamma(1+shape,scale)
    nia::T #-1/scale
end

ExpGammaIPSampler(d::Gamma) = ExpGammaIPSampler(d, GammaMTSampler)
function ExpGammaIPSampler(d::Gamma, ::Type{S}) where {S<:Sampleable}
    shape_d = shape(d)
    sampler = S(Gamma{partype(d)}(1 + shape_d, scale(d)))
    return GammaIPSampler(sampler, -inv(shape_d))
end

function rand(rng::AbstractRNG, s::ExpGammaIPSampler)
    x = log(rand(rng, s.s))
    e = randexp(rng)
    return muladd(s.nia, e, x)
end


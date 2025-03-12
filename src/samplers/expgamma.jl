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
    e = randexp(rng, typeof(x))
    return muladd(s.nia, e, x)
end


# Small Shape sampler
# From Liu, C., Martin, R., and Syring, N. (2015) for when shape < 0.3
struct ExpGammaSSSampler{T<:Real} <: Sampleable{Univariate,Continuous}
    α::T
    θ::T
    λ::T
    ω::T
    ωω::T
end

function ExpGammaSSSampler(d::Gamma)
    α = shape(d)
    ω = α / MathConstants.e / (1 - α)
    return ExpGammaSSSampler(promote(
        α,
        scale(d),
        inv(α) - 1,
        ω,
        inv(ω + 1)
    )...)
end

function rand(rng::AbstractRNG, s::ExpGammaSSSampler{T})::Float64 where T
    flT = float(T)
    while true
        U = rand(rng, flT)
        z = (U <= s.ωω) ? -log(U / s.ωω) : log(rand(rng, flT)) / s.λ
        h = exp(-z - exp(-z / s.α))
        η = z >= zero(T) ? exp(-z) : s.ω * s.λ * exp(s.λ * z)
        if h / η > rand(rng, flT)
            return s.θ - z / s.α
        end
    end
end


function _logsampler(d::Gamma)
    if shape(d) < 0.3
        return ExpGammaSSSampler(d)
    else
        return ExpGammaIPSampler(d)
    end
end

function _logrand(rng::AbstractRNG, d::Gamma)
    if shape(d) < 0.3
        return rand(rng, ExpGammaSSSampler(d))
    else
        return rand(rng, ExpGammaIPSampler(d))
    end
end

function _logrand!(rng::AbstractRNG, d::Gamma, A::AbstractArray{<:Real})
    if shape(d) < 0.3
        @inbounds for i in eachindex(A)
            A[i] = rand(rng, ExpGammaSSSampler(d))
        end
    else
        @inbounds for i in eachindex(A)
            A[i] = rand(rng, ExpGammaIPSampler(d))
        end
    end
end

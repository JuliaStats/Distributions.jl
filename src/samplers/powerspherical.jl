# Sampler for Power Spherical

struct PowerSphericalSampler{T <: Real} <: Sampleable{Multivariate,Continuous}
    μ::Vector{T}
    κ::T
    dist_b::Beta{T}
    dist_u::HyperSphericalUniform
end

function PowerSphericalSampler(μ, κ)
    d = length(μ)
    β = (d - 1) // 2
    dist_b = Beta(β + κ, β)
    dist_u = HyperSphericalUniform(d - 1)
    return PowerSphericalSampler(μ, κ, dist_b, dist_u)
end

function _rand!(rng::AbstractRNG, spl::PowerSphericalSampler, x::AbstractVector)
    v = @views x[begin+1:end]
    uhat1 = 1 - spl.μ[begin]
    μv = @views spl.μ[begin+1:end]

    z = rand(rng, spl.dist_b)
    rand!(rng, spl.dist_u, v)

    t = 2 * z - 1

    v .*= sqrt(1 - t ^ 2)

    twouuᵀy1 = t * uhat1 - dot(v, μv)
    x[begin] = t - twouuᵀy1
    v .+= μv .* (twouuᵀy1 / uhat1)
    return x
end


Base.length(s::PowerSphericalSampler) = length(s.μ)
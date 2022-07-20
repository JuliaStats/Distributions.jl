# Sampler for Power Spherical

struct PowerSphericalSampler{T <: Real} <: Sampleable{Multivariate,Continuous}
    μ::Vector{T}
    κ::T
    dist_b::Beta
    dist_u::HyperSphericalUniform
end

function _rand!(rng::AbstractRNG, spl::PowerSphericalSampler, x::AbstractVector)
    z = rand(rng, spl.dist_b)
    v = rand(rng, spl.dist_u)

    t = 2 * z - 1
    m = sqrt(1 - t ^ 2) * v'
    y = [t; m]

    x[1] = 1.
    x[2:end] .= 0.
    x .= x - spl.μ

    normalize!(x)

    x .= (-1) * (I(length(spl)) .- 2*x*x') * y
end

Base.length(s::PowerSphericalSampler) = length(s.μ)
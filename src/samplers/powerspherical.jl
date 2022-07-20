# Sampler for Power Spherical

struct PowerSphericalSampler{T <: Real} <: Sampleable{Multivariate,Continuous}
    μ::Vector{T}
    κ::T
    d::Int
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

    x .= (-1) * (I(2) .- 2*x*x') * y
end

function rand(rng::AbstractRNG, spl::PowerSphericalSampler)
    z = rand(rng, spl.dist_b)
    v = rand(rng, spl.dist_u)

    t = 2 * z - 1
    m = sqrt(1 - t ^ 2) * v'

    y = [t; m]
    e_1 = [1.; zeros(eltype(spl.μ), spl.d -1)]

    û = e_1 - spl.μ
    u = normalize(û)

    return (-1) * (I(2) .- 2*u*u') * y
end

Base.length(s::PowerSphericalSampler) = length(s.μ)
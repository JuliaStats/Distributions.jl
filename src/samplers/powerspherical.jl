# Sampler for Power Spherical

struct PowerSphericalSampler{T <: Real}
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
    e_1 = [1.; zeros(eltype(spl.μ), spl.d -1)]
    u = e_1 - spl.μ
    normalize!(u)
    x .= (-1) * (Matrix{eltype(spl.μ)}(I, spl.d, spl.d) .- 2*u*u') * y
end

function _rand(rng::AbstractRNG, spl::PowerSphericalSampler, x::AbstractVector)
    z = rand(rng, spl.dist_b)
    v = rand(rng, spl.dist_u)

    t = 2 * z - 1
    m = sqrt(1 - t ^ 2) * v'

    y = [t; m]
    e_1 = [1.; zeros(eltype(spl.μ), spl.d -1)]

    û = e_1 - spl.μ
    u = normalize(û)

    return -(Matrix{eltype(spl.μ)}(I, spl.d, spl.d) .- 2*u*u') * y
end
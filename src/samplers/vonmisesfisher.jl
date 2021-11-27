# Sampler for von Mises-Fisher
struct VonMisesFisherSampler <: Sampleable{Multivariate,Continuous}
    p::Int          # the dimension
    κ::Float64
    b::Float64
    x0::Float64
    c::Float64
    v::Vector{Float64}
end

function VonMisesFisherSampler(μ::Vector{Float64}, κ::Float64)
    p = length(μ)
    b = _vmf_bval(p, κ)
    x0 = (1.0 - b) / (1.0 + b)
    c = κ * x0 + (p - 1) * log1p(-abs2(x0))
    v = _vmf_householder_vec(μ)
    VonMisesFisherSampler(p, κ, b, x0, c, v)
end

Base.length(s::VonMisesFisherSampler) = length(s.v)

@inline function _vmf_rot!(v::AbstractVector, x::AbstractVector)
    # rotate
    scale = 2.0 * (v' * x)
    @. x -= (scale * v)
    return x
end


function _rand!(rng::AbstractRNG, spl::VonMisesFisherSampler, x::AbstractVector)
    w = _vmf_genw(rng, spl)
    p = spl.p
    x[1] = w
    s = 0.0
    @inbounds for i = 2:p
        x[i] = xi = randn(rng)
        s += abs2(xi)
    end

    # normalize x[2:p]
    r = sqrt((1.0 - abs2(w)) / s)
    @inbounds for i = 2:p
        x[i] *= r
    end

    return _vmf_rot!(spl.v, x)
end

### Core computation

_vmf_bval(p::Int, κ::Real) = (p - 1) / (2.0κ + sqrt(4 * abs2(κ) + abs2(p - 1)))

function _vmf_genw3(rng::AbstractRNG, p, b, x0, c, κ)
    ξ = rand(rng)
    w = 1.0 + (log(ξ + (1.0 - ξ)*exp(-2κ))/κ)
    return w::Float64
end

function _vmf_genwp(rng::AbstractRNG, p, b, x0, c, κ)
    r = (p - 1) / 2.0
    betad = Beta(r, r)
    z = rand(rng, betad)
    w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
    while κ * w + (p - 1) * log(1 - x0 * w) - c < log(rand(rng))
        z = rand(rng, betad)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
    end
    return w::Float64
end

# generate the W value -- the key step in simulating vMF
#
#   following movMF's document for the p != 3 case
#   and Wenzel Jakob's document for the p == 3 case
function _vmf_genw(rng::AbstractRNG, p, b, x0, c, κ)
    if p == 3
        return _vmf_genw3(rng, p, b, x0, c, κ)
    else
        return _vmf_genwp(rng, p, b, x0, c, κ)
    end
end


_vmf_genw(rng::AbstractRNG, s::VonMisesFisherSampler) =
    _vmf_genw(rng, s.p, s.b, s.x0, s.c, s.κ)

function _vmf_householder_vec(μ::Vector{Float64})
    # assuming μ is a unit-vector (which it should be)
    #  can compute v in a single pass over μ

    p = length(μ)
    v = similar(μ)
    v[1] = μ[1] - 1.0
    s = sqrt(-2*v[1])
    v[1] /= s

    @inbounds for i in 2:p
        v[i] = μ[i] / s
    end

    return v
end

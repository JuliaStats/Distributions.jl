# Sampler for Uniform Ball

struct UniformBallSampler
    n::Int
end


function _rand!(rng::AbstractRNG, spl::UniformBallSampler, x::AbstractVector)
    n = spl.n
    # defer to UniformSphericalSampler for calculation of unit-vector
    _rand!(rng, UniformSphericalSampler(n-1), x)

    # re-scale x
    u = rand(rng)
    r = (u^inv(n))
    x .*= r
    return x
end


function _rand!(rng::AbstractRNG, spl::UniformBallSampler, x::AbstractMatrix)
    for j in axes(x, 2)
        _rand!(rng, spl, view(x,:,j))
    end
    return x
end

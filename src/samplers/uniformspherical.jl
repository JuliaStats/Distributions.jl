# Sampler for Uniform Spherical

struct UniformSphericalSampler
    n::Int
end


function _rand!(rng::AbstractRNG, spl::UniformSphericalSampler, x::AbstractVector)
    n = spl.n
    s = 0.0
    @inbounds for i = 1:(n+1)
        x[i] = xi = randn(rng)
        s += abs2(xi)
    end

    # normalize x
    r = inv(sqrt(s))
    x .*= r
    return x
end


function _rand!(rng::AbstractRNG, spl::UniformSphericalSampler, x::AbstractMatrix)
    for j in axes(x, 2)
        _rand!(rng, spl, view(x,:,j))
    end
    return x
end

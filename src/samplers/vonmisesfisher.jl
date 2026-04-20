# Sampler for von Mises-Fisher
# Ref https://doi.org/10.18637/jss.v058.i10
# Ref https://hal.science/hal-04004568v3
struct VonMisesFisherSampler <: Sampleable{Multivariate,Continuous}
    p::Int          # the dimension
    κ::Float64
    b::Float64
    x0::Float64
    c::Float64
    τ::Float64
    v::Vector{Float64}
end

function VonMisesFisherSampler(μ::Vector{Float64}, κ::Float64)
    # Step 1: Calculate b, x₀, and c
    p = length(μ)
    b = (p - 1) / (2.0κ + sqrt(4 * abs2(κ) + abs2(p - 1)))
    x0 = (1.0 - b) / (1.0 + b)
    c = κ * x0 + (p - 1) * log1p(-abs2(x0))

    # Compute Householder transformation:
    # `LinearAlgebra.reflector!` computes a Householder transformation H such that
    # H μ = -copysign(|μ|₂, μ[1]) e₁
    # μ is a unit vector, and hence this implies that
    # H e₁ = μ if μ[1] < 0 and H (-e₁) = μ otherwise
    # Since `v[1] = flipsign(1, μ[1])`, the sign of `μ[1]` can be extracted from `v[1]` during sampling
    v = similar(μ)
    copyto!(v, μ)
    τ = LinearAlgebra.reflector!(v)
    
    return VonMisesFisherSampler(p, κ, b, x0, c, τ, v)
end

Base.length(s::VonMisesFisherSampler) = length(s.v)

function _rand!(rng::AbstractRNG, spl::VonMisesFisherSampler, x::AbstractVector{<:Real})
    # TODO: Generalize to more general indices
    Base.require_one_based_indexing(x)

    # Sample angle `w` assuming mean direction `(1, 0, ..., 0)`
    w = _vmf_angle(rng, spl)
    
    # Transform to sample for mean direction `(flipsign(1.0, μ[1]), 0, ..., 0)`
    v = spl.v
    w = flipsign(w, v[1])

    # Generate sample assuming mean direction `(flipsign(1.0, μ[1]), 0, ..., 0)`
    p = spl.p
    x[1] = w
    s = 0.0
    for i = 2:p
        x[i] = xi = randn(rng)
        s += abs2(xi)
    end

    # normalize x[2:p]
    r = sqrt((1.0 - abs2(w)) / s)
    for i = 2:p
        x[i] *= r
    end

    # Apply Householder transformation to mean direction `μ`
    return LinearAlgebra.reflectorApply!(v, spl.τ, x)
end

### Core computation

# Step 2: Sample angle W
function _vmf_angle(rng::AbstractRNG, spl::VonMisesFisherSampler)
    p = spl.p
    κ = spl.κ

    if p == 3
        _vmf_angle3(rng, κ)
    else
        # General case: Rejection sampling
        # Ref https://doi.org/10.18637/jss.v058.i10
        b = spl.b
        c = spl.c
        p = spl.p
        κ = spl.κ
        x0 = spl.x0
        pm1 = p - 1

        if p == 2
            # In this case the distribution reduces to the von Mises distribution on the circle
            # We exploit the fact that `Beta(1/2, 1/2) = Arcsine(0, 1)`
            dist = Arcsine(zero(b), one(b))
            while true
                z = rand(rng, dist)
                w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
                if κ * w + pm1 * log1p(- x0 * w) >= c - randexp(rng)
                    return w::Float64
                end
            end
        else
            # We sample from a `Beta((p - 1)/2, (p - 1)/2)` distribution, possibly repeatedly
            # Therefore we construct a sampler
            # To avoid the type instability of `sampler(Beta(...))` and `sampler(Gamma(...))`
            # we directly construct the Gamma sampler for Gamma((p - 1)/2, 1)
            # Since (p - 1)/2 > 1, we construct a `GammaMTSampler`
            r = pm1 / 2
            gammasampler = GammaMTSampler(Gamma{typeof(r)}(r, one(r)))
            while true
                # w is supposed to be generated as
                # z ~ Beta((p - 1)/ 2, (p - 1)/2)
                # w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
                # We sample z as 
                # z1 ~ Gamma((p - 1) / 2, 1)
                # z2 ~ Gamma((p - 1) / 2, 1)
                # z = z1 / (z1 + z2)
                # and rewrite the expression for w
                # Cf. case p == 2 above
                z1 = rand(rng, gammasampler)
                z2 = rand(rng, gammasampler)
                b_z1 = b * z1
                w = (z2 - b_z1) / (z2 + b_z1)
                if κ * w + pm1 * log1p(- x0 * w) >= c - randexp(rng)
                    return w::Float64
                end
            end
        end
    end
end

# Special case: 2-sphere
@inline function _vmf_angle3(rng::AbstractRNG, κ::Real)
    # In this case, we can directly sample the angle
    # Ref https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    ξ = rand(rng)
    w = 1.0 + (log(ξ + (1.0 - ξ)*exp(-2κ))/κ)
    return w::Float64
end

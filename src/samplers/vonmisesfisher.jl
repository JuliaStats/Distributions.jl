# Sampler for von Mises-Fisher

struct VonMisesFisherSampler
    p::Int          # the dimension
    κ::Float64
    b::Float64
    x0::Float64
    c::Float64
    Q::Matrix{Float64}
end

function VonMisesFisherSampler(μ::Vector{Float64}, κ::Float64)
    p = length(μ)
    b = _vmf_bval(p, κ)
    x0 = (1.0 - b) / (1.0 + b)
    c = κ * x0 + (p - 1) * log1p(-abs2(x0))
    Q = _vmf_rotmat(μ)
    VonMisesFisherSampler(p, κ, b, x0, c, Q)
end

function _rand!(rng::AbstractRNG, spl::VonMisesFisherSampler,
                x::AbstractVector, t::AbstractVector)
    w = _vmf_genw(rng, spl)
    p = spl.p
    t[1] = w
    s = 0.0
    for i = 2:p
        t[i] = ti = randn(rng)
        s += abs2(ti)
    end

    # normalize t[2:p]
    r = sqrt((1.0 - abs2(w)) / s)
    for i = 2:p
        t[i] *= r
    end

    # rotate
    mul!(x, spl.Q, t)
    return x
end

_rand!(rng::AbstractRNG, spl::VonMisesFisherSampler, x::AbstractVector) =
    _rand!(rng, spl, x, Vector{Float64}(undef, length(x)))

function _rand!(rng::AbstractRNG, spl::VonMisesFisherSampler, x::AbstractMatrix)
    t = Vector{Float64}(undef, size(x, 1))
    for j = 1:size(x, 2)
        _rand!(rng, spl, view(x,:,j), t)
    end
    return x
end


### Core computation

_vmf_bval(p::Int, κ::Real) = (p - 1) / (2.0κ + sqrt(4 * abs2(κ) + abs2(p - 1)))

function _vmf_genw(rng::AbstractRNG, p, b, x0, c, κ)
    # generate the W value -- the key step in simulating vMF
    #
    #   following movMF's document
    #

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

_vmf_genw(rng::AbstractRNG, s::VonMisesFisherSampler) =
    _vmf_genw(rng, s.p, s.b, s.x0, s.c, s.κ)

function _vmf_rotmat(u::Vector{Float64})
    # construct a rotation matrix Q
    # s.t. Q * [1,0,...,0]^T --> u
    #
    # Strategy: construct a full-rank matrix
    # with first column being u, and then
    # perform QR factorization
    #

    p = length(u)
    A = zeros(p, p)
    copyto!(view(A,:,1), u)

    # let k the be index of entry with max abs
    k = 1
    a = abs(u[1])
    for i = 2:p
        @inbounds ai = abs(u[i])
        if ai > a
            k = i
            a = ai
        end
    end

    # other columns of A will be filled with
    # indicator vectors, except the one
    # that activates the k-th entry
    i = 1
    for j = 2:p
        if i == k
            i += 1
        end
        A[i, j] = 1.0
    end

    # perform QR factorization
    Q = Matrix(qr!(A).Q)
    if dot(view(Q,:,1), u) < 0.0  # the first column was negated
        for i = 1:p
            @inbounds Q[i,1] = -Q[i,1]
        end
    end
    return Q
end

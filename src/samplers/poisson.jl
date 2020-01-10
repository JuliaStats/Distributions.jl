
function poissonpvec(μ::Float64, n::Int)
    # Poisson probabilities, from 0 to n
    pv = Vector{Float64}(undef, n+1)
    @inbounds pv[1] = p = exp(-μ)
    for i = 1:n
        @inbounds pv[i+1] = (p *= (μ / i))
    end
    return pv
end

# Naive sampler by counting exp variables
#
# Suitable for small μ
#
struct PoissonCountSampler{T<:Real} <: Sampleable{Univariate,Discrete}
    μ::T
end

PoissonCountSampler(d::Poisson) = PoissonCountSampler(rate(d))

function rand(rng::AbstractRNG, s::PoissonCountSampler)
    μ = s.μ
    T = typeof(μ)
    n = 0
    c = randexp(rng, T)
    while c < μ
        n += 1
        c += randexp(rng, T)
    end
    return n
end

# Algorithm from:
#
#   J.H. Ahrens, U. Dieter (1982)
#   "Computer Generation of Poisson Deviates from Modified Normal Distributions"
#   ACM Transactions on Mathematical Software, 8(2):163-179
#
#   For μ sufficiently large, (i.e. >= 10.0)
#
struct PoissonADSampler{T<:Real} <: Sampleable{Univariate,Discrete}
    μ::T
    s::T
    d::T
    L::Int
end

PoissonADSampler(d::Poisson) = PoissonADSampler(rate(d))

function PoissonADSampler(μ::Real)
    s = sqrt(μ)
    d = 6 * μ^2
    L = floor(Int, μ - 1.1484)

    PoissonADSampler(promote(μ, s, d)..., L)
end

function rand(rng::AbstractRNG, sampler::PoissonADSampler)
    μ = sampler.μ
    s = sampler.s
    d = sampler.d
    L = sampler.L
    μType = typeof(μ)

    # Step N
    G = μ + s * randn(rng, μType)

    if G >= zero(G)
        K = floor(Int, G)
        # Step I
        if K >= L
            return K
        end

        # Step S
        U = rand(rng, μType)
        if d * U >= (μ - K)^3
            return K
        end

        # Step P
        px, py, fx, fy = procf(μ, K, s)

        # Step Q
        if fy * (1 - U) <= py * exp(px - fx)
            return K
        end
    end

    while true
        # Step E
        E = randexp(rng, μType)
        U = 2 * rand(rng, μType) - one(μType)
        T = 1.8 + copysign(E, U)
        if T <= -0.6744
            continue
        end

        K = floor(Int, μ + s * T)
        px, py, fx, fy = procf(μ, K, s)
        c = 0.1069 / μ

        # Step H
        if c*abs(U) <= py*exp(px + E) - fy*exp(fx + E)
            return K
        end
    end
end

# Procedure F
function procf(μ, K::Int, s)
    # can be pre-computed, but does not seem to affect performance
    ω = 0.3989422804014327/s
    b1 = 0.041666666666666664/μ
    b2 = 0.3*b1*b1
    c3 = 0.14285714285714285*b1*b2
    c2 = b2 - 15 * c3
    c1 = b1 - 6 * b2 + 45 * c3
    c0 = 1 - b1 + 3 * b2 - 15 * c3

    if K < 10
        px = -μ
        py = μ^K/factorial(K)
    else
        δ = 0.08333333333333333/K
        δ -= 4.8*δ^3
        V = (μ-K)/K
        px = K*log1pmx(V) - δ # avoids need for table
        py = 0.3989422804014327/sqrt(K)

    end
    X = (K-μ+0.5)/s
    X2 = X^2
    fx = -X2 / 2 # missing negation in pseudo-algorithm, but appears in fortran code.
    fy = ω*(((c3*X2+c2)*X2+c1)*X2+c0)
    return px,py,fx,fy
end

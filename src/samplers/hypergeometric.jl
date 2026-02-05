"""
### Hypergeometric Sampler

***Sampling implementation reference:*** V. Kachitvichyanukul & B. Schmeiser
  "Computer generation of hypergeometric random variates"
  Journal of Statistical Computation and Simulation, 22(2):127-145
  doi:10.1080/00949658508810839
"""
struct HypergeometricSampler{T} <: Sampleable{Univariate,Discrete}

    dist::Hypergeometric
    ns_opt::Int
    nf_opt::Int
    n_opt::Int
    N::Int
    cache::T

    # some addapted names between this implementation and the original paper (see docstring)
    # | paper | here   |
    # |-------|--------|
    # | n1*   | ns     | successes in population ─┬─ stored in the distribution struct
    # | n2*   | nf     | failures in population ──┤
    # | k*    | n      | sample size ─────────────╯
    # | n1    | ns_opt | ─┬─ optimized parameters for the algorithms
    # | n2    | nf_opt | ─┤
    # | k     | n_opt  | ─╯
    # | n     | N      | total population size
    # | x     | y      | random variate being generated (specifically on HIN)

    function HypergeometricSampler(dist::Hypergeometric)

        # Step 0: Set-up constants
        N = dist.ns + dist.nf
        ns_opt, nf_opt = if dist.ns > dist.nf
            dist.nf, dist.ns
        else
            dist.ns, dist.nf
        end
        n_opt = if dist.n > N / 2
            N - dist.n
        else
            dist.n
        end
        M = floor(Int, (n_opt + 1) * (ns_opt + 1) / (N + 2))
        cache = if M - max(0, n_opt - nf_opt) < 10
            # for the HIN algorithm
            if n_opt < nf_opt
                p = exp(
                    logabsgamma(nf_opt + 1)[1] +
                    logabsgamma(N - n_opt + 1)[1] -
                    logabsgamma(N + 1)[1] -
                    logabsgamma(nf_opt - n_opt + 1)[1]
                )
                y = 0
                HINCache(p, y)
            else
                p = exp(
                    logabsgamma(ns_opt + 1)[1] +
                    logabsgamma(n_opt + 1)[1] -
                    logabsgamma(n_opt - nf_opt + 1)[1] -
                    logabsgamma(N + 1)[1]
                )
                y = n_opt - nf_opt
                HINCache(p, y)
            end
        else
            # for the H2PE algorithm
            A = logabsgamma(M + 1)[1] +
                logabsgamma(ns_opt - M + 1)[1] +
                logabsgamma(n_opt - M + 1)[1] +
                logabsgamma(nf_opt - n_opt + M + 1)[1]
            D = 1.5 * sqrt((N - n_opt) * n_opt * ns_opt * nf_opt / ((N - 1) * N * N)) + 0.5
            xL = M - D + 0.5
            xR = M + D + 0.5
            kL = exp(
                A -
                logabsgamma(xL + 1)[1] -
                logabsgamma(ns_opt - xL + 1)[1] -
                logabsgamma(n_opt - xL + 1)[1] -
                logabsgamma(nf_opt - n_opt + xL + 1)[1]
            )
            kR = exp(
                A -
                logabsgamma(xR)[1] -
                logabsgamma(ns_opt - xR + 2)[1] -
                logabsgamma(n_opt - xR + 2)[1] -
                logabsgamma(nf_opt - n_opt + xR)[1]
            )
            λL = -log(xL * (nf_opt - n_opt + xL) / ((ns_opt - xL + 1) * (n_opt - xL + 1)))
            λR = -log((ns_opt - xR + 1) * (n_opt - xR + 1) / (xR * (nf_opt - n_opt + xR)))
            p1 = 2D
            p2 = p1 + kL / λL
            p3 = p2 + kR / λR
            H2PECache(A, xL, xR, λL, λR, p1, p2, p3)
        end
        new{typeof(cache)}(dist, ns_opt, nf_opt, n_opt, N, cache)
    end

end

struct H2PECache
    A::Float64
    xL::Float64
    xR::Float64
    λL::Float64
    λR::Float64
    p1::Float64
    p2::Float64
    p3::Float64
end

struct HINCache
    p::Float64
    y::Int
end

function Random.rand(rng::AbstractRNG, spl::HypergeometricSampler{H2PECache})
    # Steps 1, 2, 3, 4: base variate generation
    cache = spl.cache
    y::Int = 0
    while true
        u = rand(rng) * cache.p3
        v = rand(rng)
        if u <= cache.p1
            # Region 1: central region
            y = floor(Int, cache.xL + u)
        elseif u <= cache.p2
            # Region 2: left exponential tail
            y = floor(Int, cache.xL + log(v) / cache.λL)
            if y < max(0, spl.n_opt - spl.nf_opt)
                continue
            end
            v = v * (u - cache.p1) * cache.λL
        else
            # Region 3: right exponential tail
            y = floor(Int, cache.xR - log(v) / cache.λR)
            if y > min(spl.ns_opt, spl.n_opt)
                continue
            end
            v = v * (u - cache.p2) * cache.λR
        end
        # Step 4: Acceptance/Regection Comparison
        # note: seems like optimizations found in the paper don't improve the
        # performance compared to this direct implementation with the lgamma function,
        # so the check is done directly for simplicity.
        if log(v) > cache.A -
                    logabsgamma(y + 1)[1] -
                    logabsgamma(spl.ns_opt - y + 1)[1] -
                    logabsgamma(spl.n_opt - y + 1)[1] -
                    logabsgamma(spl.nf_opt - spl.n_opt + y + 1)[1]
            continue
        else
            break
        end
    end

    # Step 5: return appropriate random variate
    return _hg_correct_variate(spl, y)

end

function Random.rand(rng::AbstractRNG, spl::HypergeometricSampler{HINCache})
    cache = spl.cache
    u = rand(rng)
    p = cache.p
    y = cache.y
    while u > p
        u -= p
        p *= (spl.ns_opt - y) *
             (spl.n_opt - y) /
             (y + 1) /
             (spl.nf_opt - spl.n_opt + 1 + y)
        y += 1
    end
    return _hg_correct_variate(spl, y)
end

"***Internal:*** Addapts the sampled variable from the optimized distribution for output"
function _hg_correct_variate(spl::HypergeometricSampler, y::Integer)
    dist = spl.dist
    if dist.n < spl.N / 2
        if dist.ns <= dist.nf
            y
        else
            dist.n - y
        end
    else
        if dist.ns <= dist.nf
            dist.ns - y
        else
            dist.n - dist.nf + y
        end
    end
end


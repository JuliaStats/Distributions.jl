# Helper types:

"***Internal:*** Hypergeometric distribution sampling algorithm cache"
abstract type HGSampAlgCache end


"***Internal:*** Hypergeometric distribution sampling H2PE algorithm cache"
struct H2PECache <: HGSampAlgCache
    A::Float64
    xL::Float64
    xR::Float64
    λL::Float64
    λR::Float64
    p1::Float64
    p2::Float64
    p3::Float64
end

Base.show(io::IO, cache::H2PECache) =
    print(io, "cache for H2PE sampling algorithm")

"***Internal:*** Hypergeometric distribution sampling HIN algorithm cache"
struct HINCache <: HGSampAlgCache
    p::Float64
    y::Int
end

Base.show(io::IO, cache::HINCache) =
    print(io, "cache for HIN sampling algorithm")


"""
    Hypergeometric(s, f, n)

A *Hypergeometric distribution* describes the number of successes in `n` draws without
replacement from a finite population containing `s` successes and `f` failures.

```math
P(X = k) = {{{s \\choose k} {f \\choose {n-k}}}\\over {s+f \\choose n}}, \\quad \\text{for } k = \\max(0, n - f), \\ldots, \\min(n, s).
```

```julia
Hypergeometric(s, f, n)  # Hypergeometric distribution for a population with
                         # s successes and f failures, and a sequence of n trials.

params(d)       # Get the parameters, i.e. (s, f, n)
```

External links

* [Hypergeometric distribution on Wikipedia](http://en.wikipedia.org/wiki/Hypergeometric_distribution)
* *Sampling implementation reference:* V. Kachitvichyanukul & B. Schmeiser
  "Computer generation of hypergeometric random variates"
  Journal of Statistical Computation and Simulation, 22(2):127-145
  doi:10.1080/00949658508810839
"""
struct Hypergeometric{T<:HGSampAlgCache} <: DiscreteUnivariateDistribution
    ns::Int # number of successes in population
    nf::Int # number of failures in population
    n::Int  # sample size

    ns_opt::Int # optimized parameters for sampling
    nf_opt::Int #
    n_opt::Int  #

    N::Int # = ns + nf

    cache::T # depends on the choice of sampling algorithm (H2PE or HIN)

    # some addapted names between this implementation and the original paper (see docstring)
    # | paper | here   |
    # |-------|--------|
    # | n1*   | ns     | successes in population
    # | n2*   | nf     | failures in population
    # | k*    | n      | sample size
    # | n1    | ns_opt | ─┬─ optimized parameters for the algorithms
    # | n2    | nf_opt | ─┤
    # | k     | n_opt  | ─╯
    # | n     | N      | total population size
    # | x     | y      | specifically on HIN

    function Hypergeometric(ns::Real, nf::Real, n::Real; check_args::Bool=true)

        @check_args(
            Hypergeometric,
            (ns, ns >= zero(ns)),
            (nf, nf >= zero(nf)),
            zero(n) <= n <= ns + nf,
        )

        # Step 0: Set-up constants
        N = ns + nf
        ns_opt, nf_opt = if ns > nf
            nf, ns
        else
            ns, nf
        end
        n_opt = if n > N / 2
            N - n
        else
            n
        end
        M = floor(Int, (n_opt + 1) * (ns_opt + 1) / (N + 2))
        cache = if M - max(0, n_opt - nf_opt) < 10
            # for the HIN algorithm
            if n_opt < nf_opt
                HINCache(exp(
                        logabsgamma(nf_opt + 1)[1] +
                        logabsgamma(N - n_opt + 1)[1] -
                        logabsgamma(N + 1)[1] -
                        logabsgamma(nf_opt - n_opt + 1)[1]
                    ), 0)
            else
                HINCache(exp(
                        logabsgamma(ns_opt + 1)[1] +
                        logabsgamma(n_opt + 1)[1] -
                        logabsgamma(n_opt - nf_opt + 1)[1] -
                        logabsgamma(N + 1)[1]
                    ), n_opt - nf_opt)
            end
        else
            # for the H2PE algorithm
            A = logabsgamma(M + 1)[1] +
                logabsgamma(ns_opt - M + 1)[1] +
                logabsgamma(n_opt - M + 1)[1] +
                logabsgamma(nf_opt - n_opt + M + 1)[1]
            D = 1.5 * sqrt((N - n_opt) * n * ns_opt * nf_opt / ((N - 1) * N * N)) + 0.5
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
            λL = -log(xL * (nf_opt - n_opt + xL) / ((ns_opt - xL + 1) * (n - xL + 1)))
            λR = -log((ns_opt - xR + 1) * (n_opt - xR + 1) / (xR * (nf_opt - n + xR)))
            p1 = 2D
            p2 = p1 + kL / λL
            p3 = p2 + kR / λR
            H2PECache(A, xL, xR, λL, λR, p1, p2, p3)
        end
        new{typeof(cache)}(ns, nf, n, ns_opt, nf_opt, n_opt, N, cache)
    end
end


@distr_support Hypergeometric max(d.n - d.nf, 0) min(d.ns, d.n)

partype(::Hypergeometric) = Int

### Parameters

params(d::Hypergeometric) = (d.ns, d.nf, d.n)


### Statistics

mean(d::Hypergeometric) = d.n * d.ns / (d.ns + d.nf)

function var(d::Hypergeometric)
    N = d.ns + d.nf
    p = d.ns / N
    d.n * p * (1.0 - p) * (N - d.n) / (N - 1.0)
end
mode(d::Hypergeometric) = floor(Int, (d.n + 1) * (d.ns + 1) / (d.ns + d.nf + 2))

function modes(d::Hypergeometric)
    if (d.ns == d.nf) && mod(d.n, 2) == 1
        [(d.n - 1) / 2, (d.n + 1) / 2]
    else
        [mode(d)]
    end
end

skewness(d::Hypergeometric) = (d.nf - d.ns) *
                              sqrt(d.ns + d.nf - 1) *
                              (d.ns + d.nf - 2 * d.n) /
                              sqrt(d.n * d.ns * d.nf * (d.ns + d.nf - d.n)) /
                              (d.ns + d.nf - 2)

function kurtosis(d::Hypergeometric)
    ns = float(d.ns)
    nf = float(d.nf)
    n = float(d.n)
    N = ns + nf
    a = (N - 1) * N^2 * (N * (N + 1) - 6 * ns * (N - ns) - 6 * n * (N - n)) +
        6 * n * ns * (nf) * (N - n) * (5 * N - 6)
    b = (n * ns * (N - ns) * (N - n) * (N - 2) * (N - 3))
    a / b
end

entropy(d::Hypergeometric) = entropy(map(Base.Fix1(pdf, d), support(d)))

### Evaluation & Sampling

@_delegate_statsfuns Hypergeometric hyper ns nf n

## sampling

function Random.rand(rng::AbstractRNG, dist::Hypergeometric{H2PECache})
    # Steps 1, 2, 3, 4: base variate generation
    cache = dist.cache
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
            if y < max(0, dist.n_opt - dist.nf_opt)
                continue
            end
            v = v * (u - cache.p1) * cache.λL
        else
            # Region 3: right exponential tail
            y = floor(Int, cache.xR - log(v) / cache.λR)
            if y > min(dist.ns_opt, dist.n_opt)
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
                    logabsgamma(dist.ns_opt - y + 1)[1] -
                    logabsgamma(dist.n_opt - y + 1)[1] -
                    logabsgamma(dist.nf_opt - dist.n_opt + y + 1)[1]
            continue
        else
            break
        end
    end

    # Step 5: return appropriate random variate
    return _hg_correct_variate(dist, y)

end

function Random.rand(rng::AbstractRNG, dist::Hypergeometric{HINCache})
    cache = dist.cache
    u = rand(rng)
    p = cache.p
    y = cache.y
    while u > p
        u -= p
        p *= (dist.ns_opt - y) *
             (dist.n_opt - y) /
             (y + 1) /
             (dist.nf_opt - dist.n_opt + 1 + y)
        y += 1
    end
    return _hg_correct_variate(dist, y)
end

"***Internal:*** Addapts the sampled variable from the optimized distribution for output"
function _hg_correct_variate(dist::Hypergeometric, y::Integer)
    if dist.n < dist.N / 2
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


struct HypergeometricSampler{D<:Hypergeometric} <: Sampleable{Univariate,Discrete}

    #=
    V. Kachitvichyanukul & B. Schmeiser
    "Computer generation of hypergeometric random variates"
    Journal of Statistical Computation and Simulation, 22(2):127-145
    doi:10.1080/00949658508810839
    =#

    # some addapted names between this implementation and the original paper (see docstring)
    # | paper | here     |
    # |-------|----------|
    # | n1*   | dist.ns  | successes in population ─┬─ stored in the distribution struct
    # | n2*   | dist.nf  | failures in population ──┤
    # | k*    | dist.n   | sample size ─────────────╯
    # | n1    | ns_opt   | ─┬─ optimized parameters for the algorithms
    # | n2    | nf_opt   | ─┤
    # | k     | n_opt    | ─╯
    # | n     | pop_size | total population size
    # | x     | y        | random variate being generated (specifically on HIN)

    dist::D
    ns_opt::Int
    nf_opt::Int
    n_opt::Int
    pop_size::Int

    use_HIN::Bool

    # fields for HIN
    p::Float64
    y::Int

    # fields for H2PE
    a::Float64  # named A on the paper
    xL::Float64
    xR::Float64
    λL::Float64
    λR::Float64
    p1::Float64
    p2::Float64
    p3::Float64

    function HypergeometricSampler(dist::Hypergeometric)

        # Step 0: Set-up constants
        pop_size = dist.ns + dist.nf
        ns_opt, nf_opt = minmax(dist.nf, dist.ns)
        n_opt = min(pop_size - dist.n, dist.n)
        M = fld((n_opt + 1) * (ns_opt + 1), pop_size + 2)
        p = y = a = xL = xR = λL = λR = p1 = p2 = p3 = 0.0
        use_HIN = false
        if M - max(0, n_opt - nf_opt) < 10
            use_HIN = true
            if n_opt < nf_opt
                p = exp(loggamma(nf_opt + 1) +
                        loggamma(pop_size - n_opt + 1) -
                        loggamma(pop_size + 1) -
                        loggamma(nf_opt - n_opt + 1))
                y = 0
            else
                p = exp(loggamma(ns_opt + 1) +
                        loggamma(n_opt + 1) -
                        loggamma(n_opt - nf_opt + 1) -
                        loggamma(pop_size + 1))
                y = n_opt - nf_opt
            end
        else
            # for the H2PE algorithm
            a = loggamma(M + 1) +
                loggamma(ns_opt - M + 1) +
                loggamma(n_opt - M + 1) +
                loggamma(nf_opt - n_opt + M + 1)
            D = 1.5 * sqrt((pop_size - n_opt) * n_opt * ns_opt * nf_opt /
                           ((pop_size - 1) * pop_size * pop_size)) + 0.5
            xL = M - D + 0.5
            xR = M + D + 0.5
            kL = exp(a - loggamma(xL + 1) -
                     loggamma(ns_opt - xL + 1) -
                     loggamma(n_opt - xL + 1) -
                     loggamma(nf_opt - n_opt + xL + 1))
            kR = exp(a - loggamma(xR) -
                     loggamma(ns_opt - xR + 2) -
                     loggamma(n_opt - xR + 2) -
                     loggamma(nf_opt - n_opt + xR))
            λL = -log(xL * (nf_opt - n_opt + xL) / ((ns_opt - xL + 1) * (n_opt - xL + 1)))
            λR = -log((ns_opt - xR + 1) * (n_opt - xR + 1) / (xR * (nf_opt - n_opt + xR)))
            p1 = 2D
            p2 = p1 + kL / λL
            p3 = p2 + kR / λR
        end
        new{typeof(dist)}(dist, ns_opt, nf_opt, n_opt, pop_size, use_HIN,
            p, y, a, xL, xR, λL, λR, p1, p2, p3)
    end

end


function Random.rand(rng::AbstractRNG, spl::HypergeometricSampler)
    y = 0
    if spl.use_HIN
        u = rand(rng)
        p = spl.p
        y = spl.y
        while u > p
            u -= p
            p *= (spl.ns_opt - y) *
                 (spl.n_opt - y) /
                 (y + 1) /
                 (spl.nf_opt - spl.n_opt + 1 + y)
            y += 1
        end
    else # use H2PE:
        # Steps 1, 2, 3, 4: base variate generation
        spl = spl.cache
        y::Int = 0
        while true
            u = rand(rng) * spl.p3
            v = rand(rng)
            if u <= spl.p1
                # Region 1: central region
                y = floor(Int, spl.xL + u)
            elseif u <= spl.p2
                # Region 2: left exponential tail
                y = floor(Int, spl.xL + log(v) / spl.λL)
                if y < max(0, spl.n_opt - spl.nf_opt)
                    continue
                end
                v = v * (u - spl.p1) * spl.λL
            else
                # Region 3: right exponential tail
                y = floor(Int, spl.xR - log(v) / spl.λR)
                if y > min(spl.ns_opt, spl.n_opt)
                    continue
                end
                v = v * (u - spl.p2) * spl.λR
            end
            # Step 4: Acceptance/Regection Comparison
            # note: seems like optimizations found in the paper don't improve the
            # performance compared to this direct implementation with the lgamma function,
            # so the check is done directly for simplicity.
            if log(v) > spl.a -
                        loggamma(y + 1) -
                        loggamma(spl.ns_opt - y + 1) -
                        loggamma(spl.n_opt - y + 1) -
                        loggamma(spl.nf_opt - spl.n_opt + y + 1)
                continue
            else
                break
            end
        end

        # Step 5: return appropriate random variate
        return _hg_correct_variate(spl, y)
    end

end


# Adapt the sampled variable from the optimized distribution for output
function _hg_correct_variate(spl::HypergeometricSampler, y::Integer)
    dist = spl.dist
    if dist.n < spl.pop_size / 2
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


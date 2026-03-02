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

    # fields for H2PE
    m::Int      # named M on the paper
    a::Float64  # named A on the paper
    xL::Float64
    xR::Float64
    λL::Float64
    λR::Float64
    p1::Float64
    p2::Float64
    p3::Float64

    # fields for HIN reuse fields from H2PE
    # m is y
    # a is p

    function HypergeometricSampler(dist::Hypergeometric)

        # Step 0: Set-up constants
        pop_size = dist.ns + dist.nf
        ns_opt, nf_opt = minmax(dist.nf, dist.ns)
        n_opt = min(pop_size - dist.n, dist.n)
        m = fld((n_opt + 1) * (ns_opt + 1), pop_size + 2)
        a = xL = xR = λL = λR = p1 = p2 = p3 = 0.0
        use_HIN = m < 10 + max(0, n_opt - nf_opt)
        if use_HIN
            # m is y, a is p
            if n_opt < nf_opt
                a = exp(loggamma(nf_opt + 1) +
                        loggamma(pop_size - n_opt + 1) -
                        loggamma(pop_size + 1) -
                        loggamma(nf_opt - n_opt + 1))
                m = 0
            else
                a = exp(loggamma(ns_opt + 1) +
                        loggamma(n_opt + 1) -
                        loggamma(n_opt - nf_opt + 1) -
                        loggamma(pop_size + 1))
                m = n_opt - nf_opt
            end
        else
            # for the H2PE algorithm
            a = loggamma(m + 1) +
                loggamma(ns_opt - m + 1) +
                loggamma(n_opt - m + 1) +
                loggamma(nf_opt - n_opt + m + 1)
            D = 1.5 * sqrt((pop_size - n_opt) * n_opt * ns_opt * nf_opt /
                           ((pop_size - 1) * pop_size^2)) + 0.5
            xL = m - D + 0.5
            xR = m + D + 0.5
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
            m, a, xL, xR, λL, λR, p1, p2, p3)
    end

end


function Random.rand(rng::AbstractRNG, spl::HypergeometricSampler)
    y = 0
    (; ns_opt, nf_opt, n_opt) = spl
    if spl.use_HIN
        y = spl.m # reusing m and a fields from H2PE
        p = spl.a
        u = rand(rng)
        while u > p
            u -= p
            p *= (ns_opt - y) * (n_opt - y) / (y + 1) / (nf_opt - n_opt + 1 + y)
            y += 1
        end
    else # use H2PE:
        # Steps 1, 2, 3, 4: base variate generation
        (; m, a, xL, xR, λL, λR, p1, p2, p3) = spl
        while true
            u = rand(rng) * p3
            v = rand(rng)
            if u <= p1
                # Region 1: central region
                y = floor(Int, xL + u)
            elseif u <= p2
                # Region 2: left exponential tail
                y = floor(Int, xL + log(v) / λL)
                if y < max(0, n_opt - nf_opt)
                    continue
                end
                v = v * (u - p1) * λL
            else
                # Region 3: right exponential tail
                y = floor(Int, xR - log(v) / λR)
                if y > min(ns_opt, n_opt)
                    continue
                end
                v = v * (u - p2) * λR
            end
            # Step 4: Acceptance/Regection Comparison:
            # optimizations implemented by @oscardssmidth

            logv = log(v)
            yn = ns_opt - y + 1
            yk = n_opt - y + 1
            nk = nf_opt - n_opt + y + 1
            ymm = y - m
            RSTE = (-ymm / (y + 1), ymm / yn, ymm / yk, -ymm / nk)
            G = yn * yk / muladd(y, nk, nk) - 1

            coefs = (1.0, -0.5, 1 / 3)
            GU = G * evalpoly(G, coefs)
            GL = GU - 0.25 * G^4 / (1 + max(0.0, G))

            XMSTE = (m + 0.5, ns_opt - m + 0.5, n_opt - m + 0.5, nf_opt - n_opt + m + 0.5)
            Ub = sum(map((x, r) -> x * r * evalpoly(r, coefs), XMSTE, RSTE)) +
                 y * GU - m * GL + 0.0034
            logv > Ub && continue

            DRSTE_sum = sum(map((x, r) -> x * r^4 / (1.0 + min(r, 0.0)), XMSTE, RSTE))
            if logv < Ub - 0.25 * DRSTE_sum + (y + m) * (GL - GU) - 0.0078 ||
               logv < a -
                      loggamma(y + 1) -
                      loggamma(ns_opt - y + 1) -
                      loggamma(n_opt - y + 1) -
                      loggamma(nf_opt - n_opt + y + 1)
                break
            end
        end
    end
    return _hg_correct_variate(spl, y)
end


# Adapt the sampled variable from the optimized distribution for output
function _hg_correct_variate(spl::HypergeometricSampler, y::Integer)
    dist = spl.dist
    if 2 * dist.n < spl.pop_size
        return dist.ns <= dist.nf ? y : dist.n - y
    else
        return dist.ns <= dist.nf ? dist.ns - y : dist.n - dist.nf + y
    end
end


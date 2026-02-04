"""
    Hypergeometric(s, f, n)

A *Hypergeometric distribution* describes the number of successes in `n` draws without replacement from a finite population containing `s` successes and `f` failures.

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

"""
struct Hypergeometric <: DiscreteUnivariateDistribution
    ns::Int     # number of successes in population
    nf::Int     # number of failures in population
    n::Int      # sample size

    function Hypergeometric(ns::Real, nf::Real, n::Real; check_args::Bool=true)
        @check_args(
            Hypergeometric,
            (ns, ns >= zero(ns)),
            (nf, nf >= zero(nf)),
            zero(n) <= n <= ns + nf,
        )
        new(ns, nf, n)
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
        [(d.n-1)/2, (d.n+1)/2]
    else
        [mode(d)]
    end
end

skewness(d::Hypergeometric) = (d.nf-d.ns)*sqrt(d.ns+d.nf-1)*(d.ns+d.nf-2*d.n)/sqrt(d.n*d.ns*d.nf*(d.ns+d.nf-d.n))/(d.ns+d.nf-2)

function kurtosis(d::Hypergeometric)
    ns = float(d.ns)
    nf = float(d.nf)
    n = float(d.n)
    N = ns + nf
    a = (N-1) * N^2 * (N * (N+1) - 6*ns * (N-ns) - 6*n*(N-n)) + 6*n*ns*(nf)*(N-n)*(5*N-6)
    b = (n*ns*(N-ns) * (N-n)*(N-2)*(N-3))
    a/b
end

entropy(d::Hypergeometric) = entropy(map(Base.Fix1(pdf, d), support(d)))

### Evaluation & Sampling

@_delegate_statsfuns Hypergeometric hyper ns nf n

## sampling

# Implements
#   V. Kachitvichyanukul & B. Schmeiser
#   "Computer generation of hypergeometric random variates"
#   Journal of Statistical Computation and Simulation, 22(2):127-145
#   doi:10.1080/00949658508810839
function rand(rng::AbstractRNG, d::Hypergeometric)
    # step 0.0
    (;ns, nf, n) = d
    ntotal= ns+nf
    ns, nf = minmax(ns, nf)
    n = min(n, ntotal-n)
    y = _hyper(rng, ntotal, ns, nf, n)
    if 2*d.n < ntotal
        if d.ns <= d.nf
            return y
        else
            return  d.n - y
        end
    elseif d.ns <= d.nf
        return d.ns - y
    else
        return d.n - d.nf + y
    end
end

function _hyper(rng, ntotal, ns, nf, n)
    # step 0.1 Use HIN algorithm if fast
    M = fld((n + 1) * (ns + 1), ntotal + 2)
    if M - max(0, n-nf) < 10
        if n < nf
            p = gamma(nf + 1) * gamma(ntotal - n + 1) / (gamma(ntotal + 1) * gamma(nf - n + 1))
            x = 0
        else
            p = gamma(ns + 1) * gamma(n + 1) / (gamma(ntotal + 1) * gamma(n - nf + 1))
            x = n - nf
        end
        u = rand(rng)
        while u > p
            u -= p
            p *= (ns - x) * (n - x) / ((x + 1) * (nf - n + x + 1))
            x += 1
        end
        return x
    end
    A = loggamma(M + 1) +
        loggamma(ns - M + 1) +
        loggamma(n - M + 1) +
        loggamma(nf - n + M + 1)
    D = 1.5 * sqrt((ntotal - n) * n * ns * nf / ((ntotal-1) * ntotal^2)) + 0.5
    xL = M - D + 0.5
    xR = M + D + 0.5
    nL = exp(
        A -
        loggamma(xL + 1) -
        loggamma(ns - xL + 1) -
        loggamma(n - xL + 1) -
        loggamma(nf - n + xL + 1)
    )
    nR = exp(
        A -
        loggamma(xR) -
        loggamma(ns - xR + 2) -
        loggamma(n - xR + 2) -
        loggamma(nf - n + xR)
    )
    λL = -log(xL * (nf - n + xL) / ((ns - xL + 1) * (n - xL + 1)))
    λR = -log((ns - xR + 1) * (n - xR + 1) / (xR * (nf - n + xR)))
    p1 = 2D
    p2 = p1 + nL / λL
    p3 = p2 + nR / λR
    while true
        # Step 1:
        # Begin logic to generate hypergeometric variate y.
        # Generate u = U(0, p) for selecting the region
        # v = U(0,1) for the accept reject decision. 
        u = p3 * rand(rng)
        v = rand(rng)
        logv = log(v)
        local y::Int
        if u <= p1
            # Region 1: is selected, generate a
            # uniform variate between xL, and xR.
            y = floor(Int, xL + u)
        elseif u <= p2
            # Region 2: left exponential tail
            y = floor(Int, xL + logv / λL)
            y < max(0, n - nf) && continue
            v *= (u - p1) * λL
        else
            # Region 3. Right exponential tail
            y = floor(Int, xR - logv / λR)
            y > min(ns, n) && continue
            v *= (u - p2) * λR
        end
        # Step 4: acceptance/rejection comparison
        # TODO: as an optimization, if M<100 or y<50
        # Evaluate f (y) via recursive relationship
        # f(y+1) = f(y)(ns-y)(n-y)/((y+1)(nf-y+1))
        
        # 4.2 Squeezing: Check the value of ln(v) against upper and lower bound of ln(f(y))
        yn = ns - y + 1
        yk = n - y + 1
        nk = nf - n + y + 1
        RSTE = (y - M) ./ (-y - 1, yn, yk, -nk)
        G = yn * yk / muladd(y, nk, nk) - 1

        coefs = (0.0, 1.0, -1/2, 1/3)
        GU  = evalpoly(G, coefs)
        # use (G^2)^2 instead of G^4 since it is faster and we don't care about the tiny inaccuracy introduced
        GL = GU - 0.25 * (G^2)^2 / (1 + max(0, G))
        
        XMSTE = (M, ns - M, n - M, nf - n + M) .+ 0.5
        Ub = sum(XMSTE .* evalpoly.(RSTE, Ref(coefs))) + y * GU - M * GL + 0.0034
        logv > Ub && continue
        
        DRSTE = @. XMSTE*(RSTE^2)^2 / (1 + min(RSTE, 0.0))
        if logv < Ub - 0.25 * sum(DRSTE) + (y + M) * (GL - GU) - 0.0078
            return y
        end
        # 4.3 final rejection step
        if logv > A - loggamma(y + 1) - loggamma(ns - y + 1) - loggamma(n - y + 1) - loggamma(nf - n + y + 1)
            continue
        end
        return y
    end
end

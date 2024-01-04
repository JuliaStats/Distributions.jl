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

# TODO: remove RFunctions dependency. Implement:
#   V. Kachitvichyanukul & B. Schmeiser
#   "Computer generation of hypergeometric random variates"
#   Journal of Statistical Computation and Simulation, 22(2):127-145
#   doi:10.1080/00949658508810839
@rand_rdist(Hypergeometric)
rand(d::Hypergeometric) =
    convert(Int, StatsFuns.RFunctions.hyperrand(d.ns, d.nf, d.n))

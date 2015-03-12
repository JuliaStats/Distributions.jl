# Hypergeometric(ns, nf, n):
#
#   Consider a population with ns successes and nf failures, and
#   we draw n samples from this population without replacement,
#   the number of successes within these samples follow
#   a hyper-geometric distribution
#

immutable Hypergeometric <: DiscreteUnivariateDistribution
    ns::Int     # number of successes in population
    nf::Int     # number of failures in population
    n::Int      # sample size

    function Hypergeometric(ns::Int, nf::Int, n::Int)
        ns >= 0 || error("ns must be non-negative.")
        nf >= 0 || error("nf must be non-negative.")
        0 < n < ns + nf || error("n must have 0 < n < ns + nf")
        new(ns, nf, n)
    end

    @compat Hypergeometric(ns::Real, nf::Real, n::Real) = Hypergeometric(round(Int, ns), round(Int, nf), round(Int, n))
end

@_jl_dist_3p Hypergeometric hyper

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
    N = d.ns + d.nf
    a = (N-1) * N^2 * (N * (N+1) - 6*d.ns * (N-d.ns) - 6*d.n*(N-d.n)) + 6*d.n*d.ns*(d.nf)*(N-d.n)*(5*N-6)
    b = (d.n*d.ns*(N-d.ns) * (N-d.n)*(N-2)*(N-3))
    a/b
end


### Evaluation

immutable RecursiveHypergeomProbEvaluator <: RecursiveProbabilityEvaluator
    ns::Float64
    nf::Float64
    n::Float64
end

RecursiveHypergeomProbEvaluator(d::Hypergeometric) = RecursiveHypergeomProbEvaluator(d.ns, d.nf, d.n)

nextpdf(s::RecursiveHypergeomProbEvaluator, p::Float64, x::Integer) = 
    ((s.ns - x + 1) / x) * ((s.n - x + 1) / (s.nf - s.n + x)) * p

_pdf!(r::AbstractArray, d::Hypergeometric, rgn::UnitRange) = _pdf!(r, d, rgn, RecursiveHypergeomProbEvaluator(d))


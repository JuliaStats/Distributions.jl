immutable Hypergeometric <: DiscreteUnivariateDistribution
    ns::Int # number of successes in population
    nf::Int # number of failures in population
    n::Int  # sample size

    function Hypergeometric(ns::Int, nf::Int, n::Int)
        ns >= 0 || error("ns must be non-negative.")
        nf >= 0 || error("nf must be non-negative.")
        0 < n < ns + nf || error("n must have 0 < n < ns + nf")
        new(ns, nf, n)
    end

    # Note: `convert` ensures that the inputs are integers, otherwise it throws InexactError
    Hypergeometric(ns::Real, nf::Real, n::Real) = 
        Hypergeometric(convert(Int, ns), convert(Int, nf), convert(Int, n))
end


@_jl_dist_3p Hypergeometric hyper

# handling support
minimum(d::Hypergeometric) = max(0, d.n - d.nf)
maximum(d::Hypergeometric) = min(d.ns, d.n)
support(d::Hypergeometric) = minimum(d):maximum(d)

islowerbounded(d::Hypergeometric) = true
isupperbounded(d::Hypergeometric) = true

function _probs(d::Hypergeometric, f::Int, l::Int)
    ns = d.ns
    nf = d.nf
    n = d.n
    r = Array(Float64, l - f + 1)
    r[1] = v = pdf(d, f)
    b = f - 1
    if l > f
        for x = f+1:l
            c = ((ns - x + 1) / x) * ((n - x + 1) / (nf - n + x))
            r[x-b] = (v *= c)
        end
    end
    return r
end

probs(d::Hypergeometric) = _probs(d, minimum(d), maximum(d))

function probs(d::Hypergeometric, rgn::UnitRange)
    f, l = rgn[1], rgn[end]
    minimum(d) <= f <= l <= maximum(d) || throw(BoundsError())
    _probs(d, f, l)
end


# properties
mean(d::Hypergeometric) = d.n * d.ns / (d.ns + d.nf)

function var(d::Hypergeometric)
    N = d.ns + d.nf
    p = d.ns / N
    d.n * p * (1.0 - p) * (N - d.n) / (N - 1.0)
end
mode(d::Hypergeometric) = int(floor((d.n+1) * (d.ns+1) / (d.ns+d.nf+2)))

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


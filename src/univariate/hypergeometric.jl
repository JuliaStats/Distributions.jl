immutable HyperGeometric <: DiscreteUnivariateDistribution
    ns::Float64 # number of successes in population
    nf::Float64 # number of failures in population
    n::Float64  # sample size
    function HyperGeometric(s::Real, f::Real, n::Real)
        isinteger(s) && zero(s) <= s || error("ns must be a non-negative integer")
        isinteger(f) && zero(f) <= f || error("nf must be a non-negative integer")
        isinteger(n) && zero(n) < n < s + f ||
            error("n must be a positive integer <= (ns + nf)")
        new(float64(s), float64(f), float64(n))
    end
end

@_jl_dist_3p HyperGeometric hyper

function insupport(d::HyperGeometric, x::Number)
    isinteger(x) && zero(x) <= x <= d.n && (d.n - d.nf) <= x <= d.ns
end

mean(d::HyperGeometric) = d.n * d.ns / (d.ns + d.nf)

function var(d::HyperGeometric)
    N = d.ns + d.nf
    p = d.ns / N
    d.n * p * (1.0 - p) * (N - d.n) / (N - 1.0)
end

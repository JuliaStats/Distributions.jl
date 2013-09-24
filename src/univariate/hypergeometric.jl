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

isupperbounded(d::Union(HyperGeometric, Type{HyperGeometric})) = true
islowerbounded(d::Union(HyperGeometric, Type{HyperGeometric})) = true
isbounded(d::Union(HyperGeometric, Type{HyperGeometric})) = true

min(d::HyperGeometric) = max(0,d.n-d.nf)
max(d::HyperGeometric) = min(d.n,d.ns)
support(d::HyperGeometric) = min(d):max(d)

insupport(d::HyperGeometric, x::Real) = isinteger(x) && zero(x) <= x <= d.n && (d.n - d.nf) <= x <= d.ns



mean(d::HyperGeometric) = d.n * d.ns / (d.ns + d.nf)

mode(d::HyperGeometric) = floor((d.n+1)*(d.ns+1)/(d.ns+d.nf+2))

function var(d::HyperGeometric)
    N = d.ns + d.nf
    d.n * (d.ns / N) * (d.nf / N) * ((N - d.n) / (N - 1.0))
end

@_jl_dist_3p HyperGeometric hyper



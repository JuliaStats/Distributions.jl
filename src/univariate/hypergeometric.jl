immutable HyperGeometric <: DiscreteUnivariateDistribution
    ns::Float64 # number of successes in population
    nf::Float64 # number of failures in population
    n::Float64  # sample size
    function HyperGeometric(s::Real, f::Real, n::Real)
        if 0.0 <= s && int(s) == s
            s = int(s)
        else
            error("ns must be a non-negative integer")
        end

        if 0.0 <= f && int(f) == f
            f = int(f)
        else
            error("nf must be a non-negative integer")
        end

        if 0.0 < n <= s + f && int(n) == n
            new(float64(s), float64(f), float64(n))
        else
            error("n must be a positive integer <= (ns + nf)")
        end
    end
end

@_jl_dist_3p HyperGeometric hyper

function insupport(d::HyperGeometric, x::Number)
	return isinteger(x) && 0.0 <= x <= d.n && (d.n - d.nf) <= x <= d.ns
end

mean(d::HyperGeometric) = d.n * d.ns / (d.ns + d.nf)

function var(d::HyperGeometric)
	N = d.ns + d.nf
	p = d.ns / N
	return d.n * p * (1.0 - p) * (N - d.n) / (N - 1.0)
end

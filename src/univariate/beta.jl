immutable Beta <: ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
    function Beta(a::Real, b::Real)
        if a > 0.0 && b > 0.0
            new(float64(a), float64(b))
        else
            error("Both alpha and beta must be positive")
        end
    end
end

Beta(a::Real) = Beta(a, a) # symmetric in [0, 1]
Beta() = Beta(1.0) # uniform

@_jl_dist_2p Beta beta

function entropy(d::Beta)
    o = lbeta(d.alpha, d.beta)
    o -= (d.alpha - 1.0) * digamma(d.alpha)
    o -= (d.beta - 1.0) * digamma(d.beta)
    o += (d.alpha + d.beta - 2.0) * digamma(d.alpha + d.beta)
    return o
end

insupport(::Beta, x::Real) = zero(x) < x < one(x)
insupport(::Type{Beta}, x::Real) = zero(x) < x < one(x)

function kurtosis(d::Beta)
    α, β = d.alpha, d.beta
    den = 6.0 * ((α - β)^2 * (α + β + 1.0) - α * β * (α + β + 2.0))
    num = α * β * (α + β + 2.0) * (α + β + 3.0)
    return den / num
end

mean(d::Beta) = d.alpha / (d.alpha + d.beta)

median(d::Beta) = quantile(d, 0.5)

function modes(d::Beta)
    if d.alpha > 1.0 && d.beta > 1.0
        return [(d.alpha - 1.0) / (d.alpha + d.beta - 2.0)]
    else
        error("Beta distribution with a <= 1 || b <= 1 has no modes")
    end
end

function rand(d::Beta)
    u = rand(Gamma(d.alpha))
    return u / (u + rand(Gamma(d.beta)))
end

# TODO: Don't create temporaries here
function rand(d::Beta, dims::Dims)
    u = rand(Gamma(d.alpha), dims)
    return u ./ (u + rand(Gamma(d.beta), dims))
end

function rand!(d::Beta, A::Array{Float64})
    for i in 1:length(A)
        A[i] = rand(d)
    end
    return A
end

function skewness(d::Beta)
    den = 2.0 * (d.beta - d.alpha) * sqrt(d.alpha + d.beta + 1.0)
    num = (d.alpha + d.beta + 2.0) * sqrt(d.alpha * d.beta)
    return den / num
end

function var(d::Beta)
    ab = d.alpha + d.beta
    return d.alpha * d.beta / (ab * ab * (ab + 1.0))
end

function fit_mle(::Type{Beta}, x::Array)
    for i in 1:length(x)
        if !insupport(Beta(), x[i])
            error("Bernoulli observations must be in [0,1]")
        end
    end
    x_bar = mean(x)
    v_bar = var(x)
    a = x_bar * (((x_bar * (1.0 - x_bar)) / v_bar) - 1.0)
    b = (1.0 - x_bar) * (((x_bar * (1.0 - x_bar)) / v_bar) - 1.0)
    return Beta(a, b)
end



##############################################################################
#
# Wishart Distribution
#
#  Parameters nu and S such that E(X) = nu * S 
#  See the rwish and dwish implementation in R's MCMCPack
#  This parametrization differs from Bernardo & Smith p 435
#  in this way: (nu, S) = (2.0 * alpha, 0.5 * beta^-1) 
#
##############################################################################

immutable Wishart <: ContinuousMatrixDistribution
    nu::Float64
    Schol::Cholesky{Float64}
    function Wishart(n::Real, Sc::Cholesky{Float64})
        if n > size(Sc, 1) - 1
            new(float64(n), Sc)
        else
            error("Wishart parameters must be df > p - 1")
        end
    end
end

Wishart(nu::Real, S::Matrix{Float64}) = Wishart(nu, cholfact(S))

function insupport(W::Wishart, X::Matrix{Float64})
    return size(X, 1) == size(X, 2) && isApproxSymmmetric(X) &&
           size(X, 1) == size(W.Schol, 1) && hasCholesky(X)
end

mean(w::Wishart) = w.nu * w.Schol[:U]' * w.Schol[:U]

pdf(W::Wishart, X::Matrix{Float64}) = exp(logpdf(W, X))

function logpdf(W::Wishart, X::Matrix{Float64})
    if !insupport(W, X)
        return -Inf
    else
        p = size(X, 1)
        logd::Float64 = W.nu * p / 2.0 * log(2.0)
        logd += W.nu / 2.0 * log(det(W.Schol))
        logd += lpgamma(p, W.nu / 2.0)
        logd = -logd
        logd += 0.5 * (W.nu - p - 1.0) * logdet(X)
        logd -= 0.5 * trace(W.Schol \ X)
        return logd
    end
end

function rand(w::Wishart)
    p = size(w.Schol, 1)
    X = zeros(p, p)
    for ii in 1:p
        X[ii, ii] = sqrt(rand(Chisq(w.nu - ii + 1)))
    end
    if p > 1
        for col in 2:p
            for row in 1:(col - 1)
                X[row, col] = randn()
            end
        end
    end
    Z = X * w.Schol[:U]
    return Z' * Z
end

var(w::Wishart) = error("Not yet implemented")

# multivariate gamma / partial gamma function
function lpgamma(p::Int64, a::Float64)
    res::Float64 = p * (p - 1.0) / 4.0 * log(pi)
    for ii in 1:p
        res += lgamma(a + (1.0 - ii) / 2.0)
    end
    return res
end

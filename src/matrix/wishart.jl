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
# This just checks if X could come from any Wishart
function insupport(::Type{Wishart}, X::Matrix{Float64})
    return size(X, 1) == size(X, 2) && isApproxSymmmetric(X) && hasCholesky(X)
end

mean(w::Wishart) = w.nu * w.Schol[:U]' * w.Schol[:U]

pdf(W::Wishart, X::Matrix{Float64}) = exp(logpdf(W, X))

dim(W::Wishart) = size(W.Schol, 1)

function expected_logdet(W::Wishart)
    logd = 0.
    d = dim(W)

    for i=1:d
        logd += digamma(0.5 * (W.nu + 1 - i))
    end

    logd += d * log(2)
    logd += logdet(W.Schol)

    return logd
end

function log_norm(W::Wishart)
    d = dim(W)
    return (W.nu / 2) * logdet(W.Schol) + (d * W.nu / 2) * log(2) + lpgamma(d, W.nu / 2)
end

function logpdf(W::Wishart, X::Matrix{Float64})
    if !insupport(W, X)
        return -Inf
    else
        d = dim(W)
        logd = -log_norm(W)
        logd += 0.5 * (W.nu - d - 1.0) * logdet(X)
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

function entropy(W::Wishart)
    d = dim(W)
    return log_norm(W) - (W.nu - d - 1) / 2 * expected_logdet(W) + W.nu * d / 2
end

var(w::Wishart) = error("Not yet implemented")

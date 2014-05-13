##############################################################################
#
# Inverse Wishart Distribution
#
#  Parameterized such that E(X) = Psi / (nu - p - 1)
#  See the riwish and diwish function of R's MCMCpack
#
##############################################################################

immutable InverseWishart <: ContinuousMatrixDistribution
    nu::Float64
    Psichol::Cholesky{Float64}
    function InverseWishart(n::Real, Pc::Cholesky{Float64})
        if n > size(Pc, 1) - 1
            new(float64(n), Pc)
        else
            error("Inverse Wishart parameters must be df > p - 1")
        end
    end
end

size(d::InverseWishart) = size(d.Psichol)
size(d::InverseWishart,n) = size(d.Psichol,n)

function InverseWishart(nu::Real, Psi::Matrix{Float64})
    InverseWishart(float64(nu), cholfact(Psi))
end

function insupport(IW::InverseWishart, X::Matrix{Float64})
    return size(X, 1) == size(X, 2) && isApproxSymmmetric(X) &&
           size(X, 1) == size(IW.Psichol, 1) && hasCholesky(X)
end
# This just checks if X could come from any Inverse-Wishart
function insupport(::Type{InverseWishart}, X::Matrix{Float64})
    return size(X, 1) == size(X, 2) && isApproxSymmmetric(X) && hasCholesky(X)
end

function mean(IW::InverseWishart)
    if IW.nu > size(IW.Psichol, 1) + 1
        return 1.0 / (IW.nu - size(IW.Psichol, 1) - 1.0) *
               IW.Psichol[:U]' * IW.Psichol[:U]
    else
        error("mean only defined for nu > p + 1")
    end
end

pdf(IW::InverseWishart, X::Matrix{Float64}) = exp(logpdf(IW, X))

function logpdf(IW::InverseWishart, X::Matrix{Float64})
    if !insupport(IW, X)
        return -Inf
    else
        p = size(X, 1)
        logd::Float64 = IW.nu * p / 2.0 * log(2.0)
        logd += lpgamma(p, IW.nu / 2.0)
        logd -= IW.nu / 2.0 * logdet(IW.Psichol)
        logd = -logd
        logd -= 0.5 * (IW.nu + p + 1.0) * logdet(X)
        logd -= 0.5 * trace(X \ (IW.Psichol[:U]' * IW.Psichol[:U]))
        return logd
    end
end

# rand(Wishart(nu, Psi^-1))^-1 is an sample from an
#  inverse wishart(nu, Psi). there is actually some wacky
#  behavior here where inv of the Cholesky returns the 
#  inverse of the original matrix, in this case we're getting
#  Psi^-1 like we want
function rand!(d::InverseWishart, X::AbstractMatrix) 
    p = size(d,1)
    size(X,1) == p && size(X,2) == p || throw(DimensionMismatch("Inconsistent argument dimensions"))
    A = zeros(p, p)
    for ii in 1:p
        A[ii, ii] = sqrt(rand(Chisq(d.nu - ii + 1)))
    end
    if p > 1
        for col in 2:p
            for row in 1:(col - 1)
                A[row, col] = randn()
            end
        end
    end
    Z = inv(Triangular(A / d.Psichol[:U],:U))
    At_mul_B!(X,Z,Z)
end

var(IW::InverseWishart) = error("Not yet implemented")

# because X == X' keeps failing due to floating point nonsense
function isApproxSymmmetric(a::Matrix{Float64})
    tmp = true
    for j in 2:size(a, 1)
        for i in 1:(j - 1)
            tmp &= abs(a[i, j] - a[j, i]) < 1e-8
        end
    end
    return tmp
end

# because isposdef keeps giving the wrong answer for samples
# from Wishart and InverseWisharts
function hasCholesky(a::Matrix{Float64})
    try achol = cholfact(a)
    catch e
        return false
    end
    return true
end

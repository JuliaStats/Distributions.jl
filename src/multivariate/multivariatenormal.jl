immutable MultivariateNormal <: ContinuousMultivariateDistribution
    mean::Vector{Float64}
    covchol::Cholesky{Float64}
    function MultivariateNormal(m::Vector{Float64}, c::Cholesky{Float64})
        if length(m) == size(c, 1) == size(c, 2)
            new(m, c)
        else
            error("Dimensions of mean vector and covariance matrix do not match")
        end
    end
end

function MultivariateNormal(mean::Vector{Float64}, cov::Matrix{Float64})
    MultivariateNormal(mean, cholfact(cov))
end
function MultivariateNormal(mean::Vector{Float64})
    MultivariateNormal(mean, eye(length(mean)))
end
function MultivariateNormal(cov::Matrix{Float64})
    MultivariateNormal(zeros(size(cov, 1)), cov)
end
MultivariateNormal() = MultivariateNormal(zeros(2), eye(2))

function cdf{T <: Real}(d::MultivariateNormal, x::Vector{T})
    k = length(d.mean)
    if k > 3
        error("Dimension larger than three is not supported yet")
    end
    stddev = sqrt(diag(var(d)))
    z = (x - d.mean) ./ stddev
    C = diagmm(d.covchol[:U], 1.0 / stddev)
    C = C'C
    if k == 3
        return tvtcdf(0, z, C[[2, 3, 6]])
    elseif k == 2
        return bvtcdf(0, z[1], z[2], C[2])
    else
        return cdf(Normal(), z[1])
    end
end

mean(d::MultivariateNormal) = d.mean

pdf{T<:Real}(d::MultivariateNormal, x::Vector{T}) = exp(logpdf(d, x))

function logpdf{T <: Real}(d::MultivariateNormal, x::Vector{T})
    k = length(d.mean)
    u = x - d.mean
    chol_ldiv!(d.covchol, u)
    return -0.5 * (k * log(2.0 * pi) + logdet(d.covchol) + dot(u, u))
end

function logpdf!{T <: Real}(r::AbstractVector,
                            d::MultivariateNormal,
                            x::Matrix{T})
    mu::Vector{Float64} = d.mean
    k = length(mu)
    if size(x, 1) != k
        throw(ArgumentError("The dimension of x is inconsistent with d."))
    end
    n = size(x, 2)
    u = Array(Float64, k, n)
    for j = 1:n # u[:,j] = x[:,j] - mu
        for i = 1:k
            u[i, j] = x[i, j] - mu[i]
        end
    end   
    chol_ldiv!(d.covchol, u)
    c::Float64 = -0.5 * (k * log(2.0 * pi) + logdet(d.covchol))
    for j = 1:n
        dot_uj = 0.0
        for i = 1:k
            dot_uj += u[i, j] * u[i, j]
        end      
        r[j] = c - 0.5 * dot_uj
    end
end

function logpdf{T <: Real}(d::MultivariateNormal, x::Matrix{T})
    r = Array(Float64, size(x, 2))
    logpdf!(r, d, x)
    return r
end

function rand(d::MultivariateNormal)
    z = randn(length(d.mean))
    return d.mean + d.covchol[:U]'z
end

function rand!(d::MultivariateNormal, X::Matrix)
    k = length(d.mean)
    m, n = size(X)
    if m != k
        throw(ArgumentError("Wrong dimensions"))
    end
    randn!(X)
    for i in 1:n
        X[:, i] = d.covchol[:U]'X[:, i] # TODO: Clean this up?
        for dim in 1:m
            X[dim, i] = X[dim, i] + d.mean[dim]
        end
    end
    return
end

var(d::MultivariateNormal) = (U = d.covchol[:U]; U'U)

function chol_ldiv!(chol::Cholesky{Float64}, u::VecOrMat{Float64})
    Base.LinAlg.LAPACK.trtrs!('U', 'T', 'N', chol.UL, u)
end

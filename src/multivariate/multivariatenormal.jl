# Multivariate Normal distribution

immutable MultivariateNormal{Cov<:AbstractPDMat} <: ContinuousMultivariateDistribution
    dim::Int
    zeromean::Bool
    μ::Vector{Float64}
    Σ::Cov
end

function MultivariateNormal{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov, zmean::Bool)
    d = length(μ)
    if dim(Σ) != d
        throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    end
    MultivariateNormal{Cov}(d, zmean, μ, Σ)
end

function MultivariateNormal{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov)
    d = length(μ)
    if dim(Σ) != d
        throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    end
    zmean::Bool = true
    for i = 1:d
        if μ[i] != 0.
            zmean = false
            break
        end
    end
    MultivariateNormal{Cov}(d, zmean, μ, Σ)
end

function MultivariateNormal{Cov<:AbstractPDMat}(Σ::Cov)
    d = dim(Σ)
    MultivariateNormal{Cov}(d, true, zeros(d), Σ)    
end

MultivariateNormal(μ::Vector{Float64}, σ::Float64) = MultivariateNormal(μ, ScalMat(length(μ), abs2(σ)))
MultivariateNormal(μ::Vector{Float64}, σ::Vector{Float64}) = MultivariateNormal(μ, PDiagMat(abs2(σ)))
MultivariateNormal(μ::Vector{Float64}, Σ::Matrix{Float64}) = MultivariateNormal(μ, PDMat(Σ))

MultivariateNormal(d::Int, σ::Float64) = MultivariateNormal(ScalMat(d, abs2(σ)))
MultivariateNormal(Σ::Matrix{Float64}) = MultivariateNormal(PDMat(Σ))

const MvNormal = MultivariateNormal


# Basic statistics

dim(d::MvNormal) = d.dim

mean(d::MvNormal) = d.μ

var(d::MvNormal) = diag(d.Σ)

cov(d::MvNormal) = full(d.Σ)

logdet_cov(d::MvNormal) = logdet(d.Σ)

mode(d::MvNormal) = d.μ

modes(d::MvNormal) = [mode(d)]

entropy(d::MvNormal) = 0.5 * (dim(d) * (float64(log2π) + 1.0) + logdet_cov(d))


# PDF evaluation

_gauss_c0(g::MvNormal) = -0.5 * (dim(g) * float64(log2π) + logdet_cov(g))

function sqmahal(d::MvNormal, x::Vector{Float64}) 
    z::Vector{Float64} = d.zeromean ? x : x - d.μ
    invquad(d.Σ, z) 
end

function sqmahal!(r::Array{Float64}, d::MvNormal, x::Matrix{Float64})
    if !(size(x, 1) == dim(d) && size(x, 2) == length(r))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    z::Matrix{Float64} = d.zeromean ? x : bsubtract(x, d.μ, 1)
    invquad!(r, d.Σ, z)
end

sqmahal(d::MvNormal, x::Matrix{Float64}) = sqmahal!(Array(Float64, size(x, 2)), d, x)

logpdf(d::MvNormal, x::Vector{Float64}) = _gauss_c0(d) - 0.5 * sqmahal(d, x) 

function logpdf!(r::Array{Float64}, d::MvNormal, x::Matrix{Float64})
    sqmahal!(r, d, x)
    c0::Float64 = _gauss_c0(d)
    for i = 1:size(x, 2)
        r[i] = c0 - 0.5 * r[i]
    end 
    r
end


# Sampling

function rand!(d::MvNormal, x::Vector{Float64})
    unwhiten!(d.Σ, randn!(x))
    if !d.zeromean
        add!(x, d.μ)
    end
    x
end

function rand!(d::MvNormal, x::Matrix{Float64})
    unwhiten!(d.Σ, randn!(x))
    if !d.zeromean
        badd!(x, d.μ, 1)
    end
    x
end


# Maximum Likelihood Estimation
#
# Specialized algorithms are respectively implemented for 
# each kind of covariance
#

function fit_mle(::Type{MvNormal{PDMat}}, x::Matrix{Float64})
    n = size(x, 2)
    mu = vec(mean(x, 2))
    z = bsubtract(x, mu, 1)
    C = Base.LinAlg.BLAS.gemm('N', 'T', 1.0/n, z, z)   
    MvNormal(mu, PDMat(C)) 
end

function fit_mle(::Type{MvNormal{PDMat}}, x::Matrix{Float64}, w::Vector{Float64})
    m = size(x, 1)
    n = size(x, 2)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions"))
    end

    inv_sw = 1.0 / sum(w)
    mu = Base.LinAlg.BLAS.gemv('N', inv_sw, x, w)

    z = Array(Float64, m, n)
    for j = 1:n
        cj = sqrt(w[j])
        for i = 1:m
            @inbounds z[i,j] = (x[i,j] - mu[i]) * cj
        end
    end
    C = Base.LinAlg.BLAS.gemm('N', 'T', inv_sw, z, z) 

    MvNormal(mu, PDMat(C))
end

function fit_mle(::Type{MvNormal{PDiagMat}}, x::Matrix{Float64})
    m = size(x, 1)
    n = size(x, 2)    

    mu = vec(mean(x, 2))
    va = zeros(Float64, m)
    for j = 1:n
        for i = 1:m
            @inbounds va[i] += abs2(x[i,j] - mu[i])
        end
    end
    multiply!(va, inv(n))

    MvNormal(mu, PDiagMat(va))
end

function fit_mle(::Type{MvNormal{PDiagMat}}, x::Matrix{Float64}, w::Vector{Float64})
    m = size(x, 1)
    n = size(x, 2)    
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions"))
    end

    inv_sw = 1.0 / sum(w)
    mu = Base.LinAlg.BLAS.gemv('N', inv_sw, x, w)

    va = zeros(Float64, m)
    for j = 1:n
        @inbounds wj = w[j]
        for i = 1:m
            @inbounds va[i] += abs2(x[i,j] - mu[i]) * wj
        end
    end
    multiply!(va, inv_sw)

    MvNormal(mu, PDiagMat(va))
end

function fit_mle(::Type{MvNormal{ScalMat}}, x::Matrix{Float64})
    m = size(x, 1)
    n = size(x, 2)    

    mu = vec(mean(x, 2))
    va = 0.
    for j = 1:n
        va_j = 0.
        for i = 1:m
            @inbounds va_j += abs2(x[i,j] - mu[i])
        end
        va += va_j
    end

    MvNormal(mu, ScalMat(m, va / (m * n)))
end

function fit_mle(::Type{MvNormal{ScalMat}}, x::Matrix{Float64}, w::Vector{Float64})
    m = size(x, 1)
    n = size(x, 2)    
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions"))
    end

    sw = sum(w)
    inv_sw = 1.0 / sw
    mu = Base.LinAlg.BLAS.gemv('N', inv_sw, x, w)

    va = 0.
    for j = 1:n
        @inbounds wj = w[j]
        va_j = 0.
        for i = 1:m
            @inbounds va_j += abs2(x[i,j] - mu[i]) * wj
        end
        va += va_j
    end

    MvNormal(mu, ScalMat(m, va / (m * sw)))
end

fit_mle(dty::Type{MvNormal}, x::Matrix{Float64}) = fit_mle(MvNormal{PDMat}, x)
fit_mle(dty::Type{MvNormal}, x::Matrix{Float64}, w::Vector{Float64}) = fit_mle(MvNormal{PDMat}, x, w)


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

function fit_mle(dty::Type{MvNormal{PDMat}}, x::Matrix{Float64}; 
    weights::Union(Vector{Float64}, Nothing) = nothing)

    n = size(x, 2)

    if weights == nothing
        n = size(x, 2)
        mu = vec(mean(x, 2))
        C = Base.LinAlg.BLAS.gemm!('N', 'T', 1.0/n, x, x, -1.0, mu * mu')
    else
        w::Vector{Float64} = weights
        if length(w) != n
            throw(ArgumentError("Inconsistent argument dimensions"))
        end

        inv_sw = 1.0 / sum(w)
        mu = Base.LinAlg.BLAS.gemv('N', inv_sw, x, w)
        z = bmultiply(x, sqrt(w), 2)
        C = Base.LinAlg.BLAS.gemm!('N', 'T', inv_sw, z, z, -1.0, mu * mu')
    end

    MvNormal(mu, PDMat(C))
end


function fit_mle(dty::Type{MvNormal{PDiagMat}}, x::Matrix{Float64}; 
    weights::Union(Vector{Float64}, Nothing) = nothing)

    d = size(x, 1)
    n = size(x, 2)

    if weights == nothing
        mu = zeros(d)
        va = zeros(d)

        for j in 1:n
            o = (j - 1) * d
            for i in 1 : d
                xi = x[o + i]
                mu[i] += xi
                va[i] += xi * xi
            end
        end

        inv_n = 1.0 / n
        for i in 1 : d
            mu[i] *= inv_n
            va[i] = va[i] * inv_n - abs2(mu[i])
        end

    else
        w::Vector{Float64} = weights
        if length(w) != n
            throw(ArgumentError("Inconsistent argument dimensions"))
        end

        inv_sw = 1.0 / sum(w)

        # mean vector
        mu = Base.LinAlg.BLAS.gemv('N', inv_sw, x, w)
        
        # variance
        va = zeros(d)
        o = 0
        for j in 1 : n
            wj = w[j]
            for i in 1 : d
                va[i] += abs2(x[o + i] - mu[i]) * wj
            end
            o += d
        end
        multiply!(va, inv_sw)
    end

    MvNormal(mu, PDiagMat(va))
end


function fit_mle(dty::Type{MvNormal{ScalMat}}, x::Matrix{Float64}; 
    weights::Union(Vector{Float64}, Nothing) = nothing)

    d = size(x, 1)
    n = size(x, 2)

    mu = zeros(d)
    inv_d = 1.0 / d

    if weights == nothing
        va = 0.
        for j in 1 : n
            vj = 0.
            for i in 1 : d
                xi = x[i, j]
                mu[i] += xi
                vj += xi * xi
            end
            va += vj * inv_d
        end
        inv_sw = 1.0 / n
    else
        w::Vector{Float64} = weights
        if length(w) != n
            throw(ArgumentError("Inconsistent argument dimensions"))
        end

        va = 0.
        sw = 0.
        o = 0
        for j in 1 : n
            vj = 0.
            wj = w[j]
            sw += wj

            for i in 1 : d
                xi = x[o + i]
                mu[i] += xi * wj
                vj += xi * xi
            end
            va += vj * inv_d * wj
            o += d
        end
        inv_sw = 1.0 / sw
    end

    su = 0.
    for i in 1 : d
        mu[i] *= inv_sw
        su += abs2(mu[i])
    end

    v = va * inv_sw - su * inv_d
    MvNormal(mu, ScalMat(d, v))
end

function fit{D<:MvNormal}(dty::Type{D}, x::Matrix{Float64}; weights::Union(Vector{Float64}, Nothing) = nothing)
    fit_mle(D, x; weights=weights)
end

function fit_mle(dty::Type{MvNormal}, x::Matrix{Float64}; weights::Union(Vector{Float64}, Nothing) = nothing)
    fit_mle(MvNormal{PDMat}, x; weights=weights)
end

function fit(dty::Type{MvNormal}, x::Matrix{Float64}; weights::Union(Vector{Float64}, Nothing) = nothing)
    fit(MvNormal{PDMat}, x; weights=weights)
end

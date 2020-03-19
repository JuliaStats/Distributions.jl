"""
    MatrixGaussian(M, Σ)
```julia
M::AbstractMatrix  m x n mean
Σ::AbstractPDMat   nm x nm covariance
```
The [matrix Gaussian distribution](https://arxiv.org/pdf/1804.11010.pdf) generalizes the multivariate normal distribution to ``m\\times n`` real matrices ``\\mathbf{X}``.
If ``\\mathbf{X}\\sim {\\cal N}(\\mathbf{M}, \\mathbf{\\Sigma})``, then its
probability density function is

```math
f(\\mathbf{X};\\mathbf{M}, \\mathbf{\\Sigma}) = \\frac{\\exp\\left( -\\frac{1}{2} \\, \\left[ \\textbf{vec}(\\mathbf{X} - \\mathbf{M})^{\\rm{T}} \\mathbf{\\Sigma}^{-1} \\textbf{vec}(\\mathbf{X} - \\mathbf{M}) \\right] \\right)}{(2\\pi)^{mn/2} |\\mathbf{\\Sigma}|^{1/2}}.
```

"""
struct MatrixGaussian{T <: Real, TM <: AbstractMatrix, TΣ <: AbstractPDMat} <: ContinuousMatrixDistribution
    M::TM
    Σ::TΣ
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function MatrixGaussian(M::AbstractMatrix{T}, Σ::AbstractPDMat{T}) where T <: Real
    m, n = size(M)
    (n*m, n*m) == size(Σ) || throw(ArgumentError("For (m x n) mean matrix M, Σ must be (mn x mn)"))
    logc0 = matrixgaussian_logc0(Σ)
    R = Base.promote_eltype(T, logc0)
    prom_M = convert(AbstractArray{R}, M)
    prom_Σ = convert(AbstractArray{R}, Σ)
    MatrixGaussian{R, typeof(prom_M), typeof(prom_Σ)}(prom_M, prom_Σ, R(logc0))
end

function MatrixGaussian(M::AbstractMatrix, Σ::AbstractPDMat)
    T = Base.promote_eltype(M, Σ)
    MatrixGaussian(convert(AbstractArray{T}, M), convert(AbstractArray{T}, Σ))
end

MatrixGaussian(M::AbstractMatrix, Σ::Union{AbstractMatrix, LinearAlgebra.Cholesky}) = MatrixGaussian(M, PDMat(Σ))

MatrixGaussian(m::Int, n::Int) = MatrixGaussian(zeros(m, n), Matrix(1.0I, m*n, m*n))

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

show(io::IO, d::MatrixGaussian) = show_multline(io, d, [(:M, d.M), (:U, Matrix(d.Σ))])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{MatrixGaussian{T}}, d::MatrixGaussian) where T <: Real
    MM = convert(AbstractArray{T}, d.M)
    ΣΣ = convert(AbstractArray{T}, d.Σ)
    MatrixGaussian{T, typeof(MM), typeof(ΣΣ)}(MM, ΣΣ, T(d.logc0))
end

function convert(::Type{MatrixGaussian{T}}, M::AbstractMatrix, Σ::AbstractPDMat, logc0) where T <: Real
    MM = convert(AbstractArray{T}, M)
    ΣΣ = convert(AbstractArray{T}, Σ)
    MatrixGaussian{T, typeof(MM), typeof(ΣΣ)}(MM, ΣΣ, T(logc0))
end

#  -----------------------------------------------------------------------------
#  Utilities for converting between S and Σ representations
#  -----------------------------------------------------------------------------

function _Σ_to_S(d::MatrixGaussian{T}) where T
    m, n = size(d)
    S = Matrix{T}(undef, m^2, n^2)
    for i in 0:m-1
        for j in 0:n-1
            S[i*m+1:(i+1)*m, j*n+1:(j+1)*n] = reshape(d.Σ[:, i + j * m + 1], (m, n))
        end
    end
    return S
end

function _S_to_Σ(S::AbstractMatrix{T}) where T
    m², n² = size(S)
    m = √m²
    n = √n²
    Σ = Matrix{T}(undef, n*m, n*m)
    for i in 0:m-1
        for j in 0:n-1
            d.Σ[:, i + j * m + 1] = vec(S[i*m+1:(i+1)*m, j*n+1:(j+1)*n])
        end
    end
    return Σ
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

size(d::MatrixGaussian) = size(d.M)

rank(d::MatrixGaussian) = minimum( size(d) )

insupport(d::MatrixGaussian, X::AbstractMatrix) = isreal(X) && size(X) == size(d)

mean(d::MatrixGaussian) = d.M

mode(d::MatrixGaussian) = d.M

cov(d::MatrixGaussian, ::Val{true}=Val(true)) = Matrix(d.Σ)

cov(d::MatrixGaussian, ::Val{false}) = Matrix(_Σ_to_S(d))

var(d::MatrixGaussian) = reshape(diag(cov(d)), size(d))

params(d::MatrixGaussian) = (d.M, d.Σ)

@inline partype(d::MatrixGaussian{T}) where {T<:Real} = T


#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function matrixgaussian_logc0(Σ::AbstractPDMat)
    mn = size(Σ)[1]
    -(mn / 2) * (logtwo + logπ) - (1 / 2) * logdet(Σ)
end

logkernel(d::MatrixGaussian, X::AbstractMatrix) = (vec(X - d.M)' * Matrix(inv(d.Σ)) * vec(X - d.M)) / -2

_logpdf(d::MatrixGaussian, X::AbstractMatrix) = logkernel(d, X) + d.logc0

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

#  https://en.wikipedia.org/wiki/Matrix_normal_distribution#Drawing_values_from_the_distribution

function _rand!(rng::AbstractRNG, d::MatrixGaussian, Y::AbstractMatrix)
    m, n = size(d)
    X = randn(rng, m*n)
    A = cholesky(d.Σ).L
    Y .= reshape(vec(d.M) + A * X, (m, n))
end

#  -----------------------------------------------------------------------------
#  Transformation
#  -----------------------------------------------------------------------------

vec(d::MatrixGaussian) = MvNormal(vec(d.M), d.Σ)

"""
    MatrixT(ν, M, Σ, Ω)

The [Matrix *t*-Distribution](https://en.wikipedia.org/wiki/Matrix_t-distribution)
generalizes the Multivariate *t*-Distribution from vectors to matrices. An
``n\\times p`` matrix ``\\mathbf{X}`` with the matrix *t*-distribution has density

```math
f(\\mathbf{X} ; \\nu,\\mathbf{M},\\boldsymbol{\\Sigma}, \\boldsymbol{\\Omega}) =
c_0 \\left|\\mathbf{I}_n + \\boldsymbol{\\Sigma}^{-1}(\\mathbf{X} - \\mathbf{M})\\boldsymbol{\\Omega}^{-1}(\\mathbf{X}-\\mathbf{M})'\\right|^{-\\frac{\\nu+n+p-1}{2}},
```

where

```math
c_0=\\frac{\\Gamma_p\\left(\\frac{\\nu+n+p-1}{2}\\right)}{(\\pi)^\\frac{np}{2} \\Gamma_p\\left(\\frac{\\nu+p-1}{2}\\right)} |\\boldsymbol{\\Omega}|^{-\\frac{n}{2}} |\\boldsymbol{\\Sigma}|^{-\\frac{p}{2}}.
```
The matrix *t*-distribution arises as the marginal distribution of ``\\mathbf{X}``
from ``p(\\mathbf{S},\\mathbf{X})=p(\\mathbf{S})p(\\mathbf{X}|\\mathbf{S})``, where

```math
\\begin{align*}
\\mathbf{S}&\\sim IW(\\nu + n - 1, \\boldsymbol{\\Sigma})\\\\
\\mathbf{X}|\\mathbf{S}&\\sim MN(\\mathbf{M}, \\mathbf{S}, \\boldsymbol{\\Omega})
\\end{align*}
```
"""
struct MatrixT{T <: Real, TM <: AbstractMatrix, ST <: AbstractPDMat} <: ContinuousMatrixDistribution

    ν::T
    M::TM
    Σ::ST
    Ω::ST

    c0::T

end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function MatrixT(ν::T, M::AbstractMatrix{T}, Σ::AbstractPDMat{T}, Ω::AbstractPDMat{T}) where T <: Real

    n, p = size(M)

    n₀ = dim(Σ)
    p₀ = dim(Ω)

    ν > 0   || throw(ArgumentError("degrees of freedom must be positive."))
    n == n₀ || throw(ArgumentError("Number of rows of M must equal dim of Σ."))
    p == p₀ || throw(ArgumentError("Number of columns of M must equal dim of Ω."))

    c₀ = _matrixt_c₀(Σ, Ω, ν)
    R = Base.promote_eltype(T, c₀)
    prom_M = convert(AbstractArray{R}, M)
    prom_Σ = convert(AbstractArray{R}, Σ)
    prom_Ω = convert(AbstractArray{R}, Ω)

    MatrixT{R, typeof(prom_M), typeof(prom_Σ)}(R(ν), prom_M, prom_Σ, prom_Ω, R(c₀))

end

function MatrixT(ν::Real, M::AbstractMatrix, Σ::AbstractPDMat, Ω::AbstractPDMat)

    T = Base.promote_eltype(ν, M, Σ, Ω)

    MatrixT(convert(T, ν), convert(AbstractArray{T}, M), convert(AbstractArray{T}, Σ), convert(AbstractArray{T}, Ω))

end

MatrixT(ν::Real, M::AbstractMatrix, Σ::AbstractMatrix,         Ω::AbstractMatrix)         = MatrixT(ν, M, PDMat(Σ), PDMat(Ω))
MatrixT(ν::Real, M::AbstractMatrix, Σ::AbstractMatrix,         Ω::AbstractPDMat)          = MatrixT(ν, M, PDMat(Σ), Ω)
MatrixT(ν::Real, M::AbstractMatrix, Σ::AbstractMatrix,         Ω::LinearAlgebra.Cholesky) = MatrixT(ν, M, PDMat(Σ), PDMat(Ω))
MatrixT(ν::Real, M::AbstractMatrix, Σ::LinearAlgebra.Cholesky, Ω::AbstractMatrix)         = MatrixT(ν, M, PDMat(Σ), PDMat(Ω))
MatrixT(ν::Real, M::AbstractMatrix, Σ::LinearAlgebra.Cholesky, Ω::AbstractPDMat)          = MatrixT(ν, M, PDMat(Σ), Ω)
MatrixT(ν::Real, M::AbstractMatrix, Σ::LinearAlgebra.Cholesky, Ω::LinearAlgebra.Cholesky) = MatrixT(ν, M, PDMat(Σ), PDMat(Ω))
MatrixT(ν::Real, M::AbstractMatrix, Σ::AbstractPDMat,          Ω::AbstractMatrix)         = MatrixT(ν, M, Σ,        PDMat(Ω))
MatrixT(ν::Real, M::AbstractMatrix, Σ::AbstractPDMat,          Ω::LinearAlgebra.Cholesky) = MatrixT(ν, M, Σ,        PDMat(Ω))

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

 show(io::IO, d::MatrixT) = show_multline(io, d, [(:ν, d.ν), (:M, d.M), (:Σ, Matrix(d.Σ)), (:Ω, Matrix(d.Ω))])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{MatrixT{T}}, d::MatrixT) where T <: Real
    MM = convert(AbstractArray{T}, d.M)
    ΣΣ = convert(AbstractArray{T}, d.Σ)
    ΩΩ = convert(AbstractArray{T}, d.Ω)
    MatrixT{T, typeof(MM), typeof(ΣΣ)}(T(d.ν), MM, ΣΣ, ΩΩ, T(d.c0))
end

function convert(::Type{MatrixT{T}}, ν, M::AbstractMatrix, Σ::AbstractPDMat, Ω::AbstractPDMat, c0) where T <: Real
    MM = convert(AbstractArray{T}, M)
    ΣΣ = convert(AbstractArray{T}, Σ)
    ΩΩ = convert(AbstractArray{T}, Ω)
    MatrixT{T, typeof(MM), typeof(ΣΣ)}(T(ν), MM, ΣΣ, ΩΩ, T(c0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

size(d::MatrixT) = size(d.M)

rank(d::MatrixT) = minimum( size(d) )

insupport(d::MatrixT, X::Matrix) = isreal(X) && size(X) == size(d)

function mean(d::MatrixT)

  n, p = size(d)

  (d.ν + p - n > 1) ? (return d.M) : throw(ArgumentError("mean only defined for df + p - n > 1"))

end

mode(d::MatrixT) = d.M

params(d::MatrixT) = (d.ν, d.M, d.Σ, d.Ω)

@inline partype(d::MatrixT{T}) where {T <: Real} = T

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function _matrixt_c₀(Σ::AbstractPDMat, Ω::AbstractPDMat, ν::Real)

    n = dim(Σ)
    p = dim(Ω)

    term₁ = logmvgamma(p, (ν + n + p - 1) / 2)
    term₂ = - (n * p / 2) * logπ
    term₃ = - logmvgamma(p, (ν + p - 1) / 2)
    term₄ = (-n / 2) * logdet(Ω)
    term₅ = (-p / 2) * logdet(Σ)

    term₁ + term₂ + term₃ + term₄ + term₅

end

function logkernel(d::MatrixT, X::AbstractMatrix)

    A  = (X - d.M)
    At = Matrix(A')

    n, p = size(d)

    (-(d.ν + n + p - 1) / 2) * logdet( I + (d.Σ \ A) * (d.Ω \ At) )

end

_logpdf(d::MatrixT, X::AbstractMatrix) = logkernel(d, X) + d.c0

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

#  Theorem 4.2.1 in Gupta and Nagar (1999)

function _rand!(rng::AbstractRNG, d::MatrixT, A::AbstractMatrix)

    n, p = size(d)

    S = rand(rng, InverseWishart(d.ν + n - 1, d.Σ) )

    A .= rand(rng, MatrixNormal(d.M, S, d.Ω) )

end

#  -----------------------------------------------------------------------------
#  Relationship with Multivariate t
#  -----------------------------------------------------------------------------

function MvTDist(MT::MatrixT)

    n, p = size(MT)

    all([n, p] .> 1) && error("Row or col dim of `MatrixT` must be 1 to coerce to `MvTDist`")

    ν, M, Σ, Ω = params(MT)

    return MvTDist(ν, vec(M), (1 / ν) * kron(Σ.mat, Ω.mat))

end

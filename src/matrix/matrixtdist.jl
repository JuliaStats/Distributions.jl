"""
    MatrixTDist(ν, M, Σ, Ω)
```julia
ν::Real            positive degrees of freedom
M::AbstractMatrix  n x p location
Σ::AbstractPDMat   n x n scale
Ω::AbstractPDMat   p x p scale
```
The [matrix *t*-Distribution](https://en.wikipedia.org/wiki/Matrix_t-distribution)
generalizes the multivariate *t*-Distribution to ``n\\times p`` real
matrices ``\\mathbf{X}``. If ``\\mathbf{X}\\sim MT_{n,p}(\\nu,\\mathbf{M},\\boldsymbol{\\Sigma},
\\boldsymbol{\\Omega})``, then its probability density function is

```math
f(\\mathbf{X} ; \\nu,\\mathbf{M},\\boldsymbol{\\Sigma}, \\boldsymbol{\\Omega}) =
c_0 \\left|\\mathbf{I}_n + \\boldsymbol{\\Sigma}^{-1}(\\mathbf{X} - \\mathbf{M})\\boldsymbol{\\Omega}^{-1}(\\mathbf{X}-\\mathbf{M})^{\\rm{T}}\\right|^{-\\frac{\\nu+n+p-1}{2}},
```

where

```math
c_0=\\frac{\\Gamma_p\\left(\\frac{\\nu+n+p-1}{2}\\right)}{(\\pi)^\\frac{np}{2} \\Gamma_p\\left(\\frac{\\nu+p-1}{2}\\right)} |\\boldsymbol{\\Omega}|^{-\\frac{n}{2}} |\\boldsymbol{\\Sigma}|^{-\\frac{p}{2}}.
```

If the joint distribution ``p(\\mathbf{S},\\mathbf{X})=p(\\mathbf{S})p(\\mathbf{X}|\\mathbf{S})``
is given by

```math
\\begin{align*}
\\mathbf{S}&\\sim IW_n(\\nu + n - 1, \\boldsymbol{\\Sigma})\\\\
\\mathbf{X}|\\mathbf{S}&\\sim MN_{n,p}(\\mathbf{M}, \\mathbf{S}, \\boldsymbol{\\Omega}),
\\end{align*}
```

then the marginal distribution of ``\\mathbf{X}`` is
``MT_{n,p}(\\nu,\\mathbf{M},\\boldsymbol{\\Sigma},\\boldsymbol{\\Omega})``.
"""
struct MatrixTDist{T <: Real, TM <: AbstractMatrix, TΣ <: AbstractPDMat, TΩ <: AbstractPDMat} <: ContinuousMatrixDistribution
    ν::T
    M::TM
    Σ::TΣ
    Ω::TΩ
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function MatrixTDist(ν::T, M::AbstractMatrix{T}, Σ::AbstractPDMat{T}, Ω::AbstractPDMat{T}) where T <: Real
    n, p = size(M)
    0 < ν < Inf || throw(ArgumentError("degrees of freedom must be positive and finite."))
    n == dim(Σ) || throw(ArgumentError("Number of rows of M must equal dim of Σ."))
    p == dim(Ω) || throw(ArgumentError("Number of columns of M must equal dim of Ω."))
    logc0 = matrixtdist_logc0(Σ, Ω, ν)
    R = Base.promote_eltype(T, logc0)
    prom_M = convert(AbstractArray{R}, M)
    prom_Σ = convert(AbstractArray{R}, Σ)
    prom_Ω = convert(AbstractArray{R}, Ω)
    MatrixTDist{R, typeof(prom_M), typeof(prom_Σ), typeof(prom_Ω)}(R(ν), prom_M, prom_Σ, prom_Ω, R(logc0))
end

function MatrixTDist(ν::Real, M::AbstractMatrix, Σ::AbstractPDMat, Ω::AbstractPDMat)
    T = Base.promote_eltype(ν, M, Σ, Ω)
    MatrixTDist(convert(T, ν), convert(AbstractArray{T}, M), convert(AbstractArray{T}, Σ), convert(AbstractArray{T}, Ω))
end

MatrixTDist(ν::Real, M::AbstractMatrix, Σ::Union{AbstractMatrix, LinearAlgebra.Cholesky}, Ω::Union{AbstractMatrix, LinearAlgebra.Cholesky}) = MatrixTDist(ν, M, PDMat(Σ), PDMat(Ω))
MatrixTDist(ν::Real, M::AbstractMatrix, Σ::AbstractPDMat, Ω::Union{AbstractMatrix, LinearAlgebra.Cholesky}) = MatrixTDist(ν, M, Σ, PDMat(Ω))
MatrixTDist(ν::Real, M::AbstractMatrix, Σ::Union{AbstractMatrix, LinearAlgebra.Cholesky}, Ω::AbstractPDMat) = MatrixTDist(ν, M, PDMat(Σ), Ω)

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

 show(io::IO, d::MatrixTDist) = show_multline(io, d, [(:ν, d.ν), (:M, d.M), (:Σ, Matrix(d.Σ)), (:Ω, Matrix(d.Ω))])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{MatrixTDist{T}}, d::MatrixTDist) where T <: Real
    MM = convert(AbstractArray{T}, d.M)
    ΣΣ = convert(AbstractArray{T}, d.Σ)
    ΩΩ = convert(AbstractArray{T}, d.Ω)
    MatrixTDist{T, typeof(MM), typeof(ΣΣ), typeof(ΩΩ)}(T(d.ν), MM, ΣΣ, ΩΩ, T(d.logc0))
end

function convert(::Type{MatrixTDist{T}}, ν, M::AbstractMatrix, Σ::AbstractPDMat, Ω::AbstractPDMat, logc0) where T <: Real
    MM = convert(AbstractArray{T}, M)
    ΣΣ = convert(AbstractArray{T}, Σ)
    ΩΩ = convert(AbstractArray{T}, Ω)
    MatrixTDist{T, typeof(MM), typeof(ΣΣ), typeof(ΩΩ)}(T(ν), MM, ΣΣ, ΩΩ, T(logc0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

size(d::MatrixTDist) = size(d.M)

rank(d::MatrixTDist) = minimum( size(d) )

insupport(d::MatrixTDist, X::Matrix) = isreal(X) && size(X) == size(d)

function mean(d::MatrixTDist)
    n, p = size(d)
    d.ν + p - n > 1 || throw(ArgumentError("mean only defined for df + p - n > 1"))
    return d.M
end

mode(d::MatrixTDist) = d.M

cov(d::MatrixTDist, ::Val{true}=Val(true)) = d.ν <= 2 ? throw(ArgumentError("cov only defined for df > 2")) : Matrix(kron(d.Ω, d.Σ)) ./ (d.ν - 2)

cov(d::MatrixTDist, ::Val{false}) = ((n, p) = size(d); reshape(cov(d), n, p, n, p))

var(d::MatrixTDist) = d.ν <= 2 ? throw(ArgumentError("var only defined for df > 2")) : reshape(diag(cov(d)), size(d))

params(d::MatrixTDist) = (d.ν, d.M, d.Σ, d.Ω)

@inline partype(d::MatrixTDist{T}) where {T <: Real} = T

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function matrixtdist_logc0(Σ::AbstractPDMat, Ω::AbstractPDMat, ν::Real)
    #  returns the natural log of the normalizing constant for the pdf
    n = dim(Σ)
    p = dim(Ω)
    term1 = logmvgamma(p, (ν + n + p - 1) / 2)
    term2 = - (n * p / 2) * logπ
    term3 = - logmvgamma(p, (ν + p - 1) / 2)
    term4 = (-n / 2) * logdet(Ω)
    term5 = (-p / 2) * logdet(Σ)
    term1 + term2 + term3 + term4 + term5
end

function logkernel(d::MatrixTDist, X::AbstractMatrix)
    n, p = size(d)
    A = X - d.M
    (-(d.ν + n + p - 1) / 2) * logdet( I + (d.Σ \ A) * (d.Ω \ A') )
end

_logpdf(d::MatrixTDist, X::AbstractMatrix) = logkernel(d, X) + d.logc0

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

#  Theorem 4.2.1 in Gupta and Nagar (1999)

function _rand!(rng::AbstractRNG, d::MatrixTDist, A::AbstractMatrix)
    n, p = size(d)
    S = rand(rng, InverseWishart(d.ν + n - 1, d.Σ) )
    A .= rand(rng, MatrixNormal(d.M, S, d.Ω) )
end

#  -----------------------------------------------------------------------------
#  Relationship with Multivariate t
#  -----------------------------------------------------------------------------
#  if a t-distributed random matrix is in fact just a row or column,
#  it is equivalent to a t-distributed random vector.
#  -----------------------------------------------------------------------------

function MvTDist(MT::MatrixTDist)
    n, p = size(MT)
    all([n, p] .> 1) && error("Row or col dim of `MatrixTDist` must be 1 to coerce to `MvTDist`")
    ν, M, Σ, Ω = params(MT)
    MvTDist(ν, vec(M), (1 / ν) * kron(Σ, Ω))
end

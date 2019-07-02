"""
    MatrixNormal{T <: Real, TM <: AbstractMatrix, ST <: AbstractPDMat} <: ContinuousMatrixDistribution

[Matrix Normal Distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution)

`X ~ MN(M, U, V)`

M: n x p

U: n x n positive definite

V: p x p positive definite

f(X) = c0 * exp( -0.5 tr[inv(V) (X - M)' inv(U) (X - M)] )

c0   = (2pi) ^ {-np / 2} |V| ^ {-n / 2} |U| ^ {-p / 2}

"""
struct MatrixNormal{T <: Real, TM <: AbstractMatrix, ST <: AbstractPDMat} <: ContinuousMatrixDistribution

    M::TM
    U::ST
    V::ST

    c0::T

end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function MatrixNormal(M::AbstractMatrix{T}, U::AbstractPDMat{T}, V::AbstractPDMat{T}) where T <: Real

    n = size(M, 1)
    p = size(M, 2)

    n₀ = size(U, 1)
    p₀ = size(V, 1)

    n != n₀ && error("Number of rows of M must equal dim of U.")
    p != p₀ && error("Number of columns of M must equal dim of V.")

    c₀ = _matrixnormal_c₀(U, V)

    R = Base.promote_eltype(T, c₀)

    prom_M = convert(AbstractArray{R}, M)
    prom_U = convert(AbstractArray{R}, U)
    prom_V = convert(AbstractArray{R}, V)

    MatrixNormal{R, typeof(prom_M), typeof(prom_U)}(prom_M, prom_U, prom_V, R(c₀))

end

function MatrixNormal(M::AbstractMatrix, U::AbstractPDMat, V::AbstractPDMat)

    T = Base.promote_eltype(M, U, V)

    MatrixNormal(convert(AbstractArray{T}, M), convert(AbstractArray{T}, U), convert(AbstractArray{T}, V))

end

MatrixNormal(M::AbstractMatrix, U::AbstractMatrix,         V::AbstractMatrix)         = MatrixNormal(M, PDMat(U), PDMat(V))
MatrixNormal(M::AbstractMatrix, U::AbstractMatrix,         V::LinearAlgebra.Cholesky) = MatrixNormal(M, PDMat(U), PDMat(V))
MatrixNormal(M::AbstractMatrix, U::AbstractMatrix,         V::AbstractPDMat)          = MatrixNormal(M, PDMat(U), V)
MatrixNormal(M::AbstractMatrix, U::LinearAlgebra.Cholesky, V::AbstractMatrix)         = MatrixNormal(M, PDMat(U), PDMat(V))
MatrixNormal(M::AbstractMatrix, U::LinearAlgebra.Cholesky, V::LinearAlgebra.Cholesky) = MatrixNormal(M, PDMat(U), PDMat(V))
MatrixNormal(M::AbstractMatrix, U::LinearAlgebra.Cholesky, V::AbstractPDMat)          = MatrixNormal(M, PDMat(U), V)
MatrixNormal(M::AbstractMatrix, U::AbstractPDMat,          V::AbstractMatrix)         = MatrixNormal(M, U,        PDMat(V))
MatrixNormal(M::AbstractMatrix, U::AbstractPDMat,          V::LinearAlgebra.Cholesky) = MatrixNormal(M, U,        PDMat(V))

MatrixNormal(m::Int, n::Int) = MatrixNormal(zeros(m, n), Matrix(1.0I, m, m), Matrix(1.0I, n, n))

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

show(io::IO, d::MatrixNormal) = show_multline(io, d, [(:M, d.M), (:U, Matrix(d.U)), (:V, Matrix(d.V))])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{MatrixNormal{T}}, d::MatrixNormal) where T <: Real
    MM = convert(AbstractArray{T}, d.M)
    UU = convert(AbstractArray{T}, d.U)
    VV = convert(AbstractArray{T}, d.V)
    MatrixNormal{T, typeof(MM), typeof(UU)}(MM, UU, VV, T(d.c0))
end

function convert(::Type{MatrixNormal{T}}, M::AbstractMatrix, U::AbstractPDMat, V::AbstractPDMat, c0) where T <: Real
    MM = convert(AbstractArray{T}, M)
    UU = convert(AbstractArray{T}, U)
    VV = convert(AbstractArray{T}, V)
    MatrixNormal{T, typeof(MM), typeof(UU)}(MM, UU, VV, T(c0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

size(d::MatrixNormal) = size(d.M)

rank(d::MatrixNormal) = minimum( size(d) )

insupport(d::MatrixNormal, X::Matrix) = isreal(X) && size(X) == size(d)

mean(d::MatrixNormal) = d.M

mode(d::MatrixNormal) = d.M

params(d::MatrixNormal) = (d.M, d.U, d.V)

@inline partype(d::MatrixNormal{T}) where {T<:Real} = T

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function _matrixnormal_c₀(U::AbstractPDMat, V::AbstractPDMat)

    n = dim(U)
    p = dim(V)

    -(n * p / 2) * (logtwo + logπ) - (n / 2) * logdet(V) - (p / 2) * logdet(U)

end

function logkernel(d::MatrixNormal, X::AbstractMatrix)

    A  = X - d.M
    At = Matrix(A')

    -0.5 * tr( (d.V \ At) * (d.U \ A) )

end

_logpdf(d::MatrixNormal, X::AbstractMatrix) = logkernel(d, X) + d.c0

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

function _rand!(rng::AbstractRNG, d::MatrixNormal, A::AbstractMatrix)

    n, p = size(d)

    X = randn(rng, n, p)

    A .= d.M + d.U.chol.L * X * d.V.chol.U

end

#  -----------------------------------------------------------------------------
#  Transformation
#  -----------------------------------------------------------------------------

vec(d::MatrixNormal) = MvNormal(vec(d.M), kron(d.V.mat, d.U.mat))

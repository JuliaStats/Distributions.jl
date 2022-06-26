"""
    MatrixNormal(M, U, V)
```julia
M::AbstractMatrix  n x p mean
U::AbstractPDMat   n x n row covariance
V::AbstractPDMat   p x p column covariance
```
The [matrix normal distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution)
generalizes the multivariate normal distribution to ``n\\times p`` real matrices ``\\mathbf{X}``.
If ``\\mathbf{X}\\sim \\textrm{MN}_{n,p}(\\mathbf{M}, \\mathbf{U}, \\mathbf{V})``, then its
probability density function is

```math
f(\\mathbf{X};\\mathbf{M}, \\mathbf{U}, \\mathbf{V}) = \\frac{\\exp\\left( -\\frac{1}{2} \\, \\mathrm{tr}\\left[ \\mathbf{V}^{-1} (\\mathbf{X} - \\mathbf{M})^{\\rm{T}} \\mathbf{U}^{-1} (\\mathbf{X} - \\mathbf{M}) \\right] \\right)}{(2\\pi)^{np/2} |\\mathbf{V}|^{n/2} |\\mathbf{U}|^{p/2}}.
```

``\\mathbf{X}\\sim \\textrm{MN}_{n,p}(\\mathbf{M},\\mathbf{U},\\mathbf{V})``
if and only if ``\\text{vec}(\\mathbf{X})\\sim \\textrm{N}(\\text{vec}(\\mathbf{M}),\\mathbf{V}\\otimes\\mathbf{U})``.
"""
struct MatrixNormal{T <: Real, TM <: AbstractMatrix, TU <: AbstractPDMat, TV <: AbstractPDMat} <: ContinuousMatrixDistribution
    M::TM
    U::TU
    V::TV
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function MatrixNormal(M::AbstractMatrix{T}, U::AbstractPDMat{T}, V::AbstractPDMat{T}) where T <: Real
    n, p = size(M)
    n == size(U, 1) || throw(ArgumentError("Number of rows of M must equal dim of U."))
    p == size(V, 1) || throw(ArgumentError("Number of columns of M must equal dim of V."))
    logc0 = matrixnormal_logc0(U, V)
    R = Base.promote_eltype(T, logc0)
    prom_M = convert(AbstractArray{R}, M)
    prom_U = convert(AbstractArray{R}, U)
    prom_V = convert(AbstractArray{R}, V)
    MatrixNormal{R, typeof(prom_M), typeof(prom_U), typeof(prom_V)}(prom_M, prom_U, prom_V, R(logc0))
end

function MatrixNormal(M::AbstractMatrix, U::AbstractPDMat, V::AbstractPDMat)
    T = Base.promote_eltype(M, U, V)
    MatrixNormal(convert(AbstractArray{T}, M), convert(AbstractArray{T}, U), convert(AbstractArray{T}, V))
end

MatrixNormal(M::AbstractMatrix, U::Union{AbstractMatrix, LinearAlgebra.Cholesky}, V::Union{AbstractMatrix, LinearAlgebra.Cholesky}) = MatrixNormal(M, PDMat(U), PDMat(V))
MatrixNormal(M::AbstractMatrix, U::Union{AbstractMatrix, LinearAlgebra.Cholesky}, V::AbstractPDMat) = MatrixNormal(M, PDMat(U), V)
MatrixNormal(M::AbstractMatrix, U::AbstractPDMat, V::Union{AbstractMatrix, LinearAlgebra.Cholesky}) = MatrixNormal(M, U, PDMat(V))

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
    MatrixNormal{T, typeof(MM), typeof(UU), typeof(VV)}(MM, UU, VV, T(d.logc0))
end
Base.convert(::Type{MatrixNormal{T}}, d::MatrixNormal{T}) where {T<:Real} = d

function convert(::Type{MatrixNormal{T}}, M::AbstractMatrix, U::AbstractPDMat, V::AbstractPDMat, logc0) where T <: Real
    MM = convert(AbstractArray{T}, M)
    UU = convert(AbstractArray{T}, U)
    VV = convert(AbstractArray{T}, V)
    MatrixNormal{T, typeof(MM), typeof(UU), typeof(VV)}(MM, UU, VV, T(logc0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

size(d::MatrixNormal) = size(d.M)

rank(d::MatrixNormal) = minimum( size(d) )

insupport(d::MatrixNormal, X::AbstractMatrix) = isreal(X) && size(X) == size(d)

mean(d::MatrixNormal) = d.M

mode(d::MatrixNormal) = d.M

cov(d::MatrixNormal) = Matrix(kron(d.V, d.U))

cov(d::MatrixNormal, ::Val{false}) = ((n, p) = size(d); reshape(cov(d), n, p, n, p))

var(d::MatrixNormal) = reshape(diag(cov(d)), size(d))

params(d::MatrixNormal) = (d.M, d.U, d.V)

@inline partype(d::MatrixNormal{T}) where {T<:Real} = T

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function matrixnormal_logc0(U::AbstractPDMat, V::AbstractPDMat)
    n = size(U, 1)
    p = size(V, 1)
    -(n * p / 2) * (logtwo + logπ) - (n / 2) * logdet(V) - (p / 2) * logdet(U)
end

function logkernel(d::MatrixNormal, X::AbstractMatrix)
    A  = X - d.M
    -0.5 * tr( (d.V \ A') * (d.U \ A) )
end

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

#  https://en.wikipedia.org/wiki/Matrix_normal_distribution#Drawing_values_from_the_distribution

function _rand!(rng::AbstractRNG, d::MatrixNormal, Y::AbstractMatrix)
    n, p = size(d)
    X = randn(rng, n, p)
    A = cholesky(d.U).L
    B = cholesky(d.V).U
    Y .= d.M .+ A * X * B
end

#  -----------------------------------------------------------------------------
#  Test utils
#  -----------------------------------------------------------------------------

function _univariate(d::MatrixNormal)
    check_univariate(d)
    M, U, V = params(d)
    μ = M[1]
    σ = sqrt( Matrix(U)[1] * Matrix(V)[1] )
    return Normal(μ, σ)
end

function _multivariate(d::MatrixNormal)
    n, p = size(d)
    all([n, p] .> 1) && throw(ArgumentError("Row or col dim of `MatrixNormal` must be 1 to coerce to `MvNormal`"))
    return vec(d)
end

function _rand_params(::Type{MatrixNormal}, elty, n::Int, p::Int)
    M = randn(elty, n, p)
    U = (X = 2 .* rand(elty, n, n) .- 1; X * X')
    V = (Y = 2 .* rand(elty, p, p) .- 1; Y * Y')
    return M, U, V
end

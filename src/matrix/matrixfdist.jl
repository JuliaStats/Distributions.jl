"""
    MatrixFDist(n1, n2, B)
```julia
n1::Real          degrees of freedom (greater than p - 1)
n2::Real          degrees of freedom (greater than p - 1)
B::AbstractPDMat  p x p scale
```
The [matrix *F*-Distribution](https://projecteuclid.org/euclid.ba/1515747744)
(sometimes called the matrix beta type II distribution) generalizes the
*F*-Distribution to ``p\\times p`` real, positive definite matrices ``\\boldsymbol{\\Sigma}``.
If ``\\boldsymbol{\\Sigma}\\sim MF_{p}(n_1/2,n_2/2,\\mathbf{B})``, then its probability density function is

```math
f(\\boldsymbol{\\Sigma} ; n_1,n_2,\\mathbf{B}) =
\\frac{\\Gamma_p(\\frac{n_1+n_2}{2})}{\\Gamma_p(\\frac{n_1}{2})\\Gamma_p(\\frac{n_2}{2})}
|\\mathbf{B}|^{n_2/2}|\\boldsymbol{\\Sigma}|^{(n_1-p-1)/2}|\\mathbf{B}+\\boldsymbol{\\Sigma}|^{-(n_1+n_2)/2}.
```

If the joint distribution ``p(\\boldsymbol{\\Psi},\\boldsymbol{\\Sigma})=p(\\boldsymbol{\\Psi})p(\\boldsymbol{\\Sigma}|\\boldsymbol{\\Psi})``
is given by

```math
\\begin{align*}
\\boldsymbol{\\Psi}&\\sim W_p(n_1, \\mathbf{B})\\\\
\\boldsymbol{\\Sigma}|\\boldsymbol{\\Psi}&\\sim IW_p(n_2, \\boldsymbol{\\Psi}),
\\end{align*}
```

then the marginal distribution of ``\\boldsymbol{\\Sigma}`` is
``MF_{p}(n_1/2,n_2/2,\\mathbf{B})``.
"""
struct MatrixFDist{T <: Real, TW <: Wishart} <: ContinuousMatrixDistribution
    W::TW
    n2::T
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function MatrixFDist(n1::Real, n2::Real, B::AbstractPDMat)
    p = dim(B)
    n1 > p - 1 || throw(ArgumentError("first degrees of freedom must be larger than $(p - 1)"))
    n2 > p - 1 || throw(ArgumentError("second degrees of freedom must be larger than $(p - 1)"))
    logc0 = matrixfdist_logc0(n1, n2, B)
    T = Base.promote_eltype(n1, n2, logc0, B)
    prom_B = convert(AbstractArray{T}, B)
    W = Wishart(T(n1), prom_B)
    MatrixFDist{T, typeof(W)}(W, T(n2), T(logc0))
end

MatrixFDist(n1::Real, n2::Real, B::Union{AbstractMatrix, LinearAlgebra.Cholesky}) = MatrixFDist(n1, n2, PDMat(B))

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

 show(io::IO, d::MatrixFDist) = show_multline(io, d, [(:n1, d.W.df), (:n2, d.n2), (:B, Matrix(d.W.S))])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{MatrixFDist{T}}, d::MatrixFDist) where T <: Real
    W = convert(Wishart{T}, d.W)
    MatrixFDist{T, typeof(W)}(W, T(d.n2), T(d.logc0))
end

function convert(::Type{MatrixFDist{T}}, W::Wishart, n2, logc0) where T <: Real
    WW = convert(Wishart{T}, W)
    MatrixFDist{T, typeof(WW)}(WW, T(n2), T(logc0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

dim(d::MatrixFDist) = dim(d.W)

size(d::MatrixFDist) = size(d.W)

rank(d::MatrixFDist) = dim(d)

insupport(d::MatrixFDist, Σ::AbstractMatrix) = isreal(Σ) && size(Σ) == size(d) && isposdef(Σ)

params(d::MatrixFDist) = (d.W.df, d.n2, d.W.S)

function mean(d::MatrixFDist)
    p = dim(d)
    n1, n2, B = params(d)
    n2 > p + 1 || throw(ArgumentError("mean only defined for df2 > dim + 1"))
    return (n1 / (n2 - p - 1)) * Matrix(B)
end

@inline partype(d::MatrixFDist{T}) where {T <: Real} = T

#  Konno (1988 JJSS) Corollary 2.4.i
function cov(d::MatrixFDist, i::Integer, j::Integer, k::Integer, l::Integer)
    p = dim(d)
    n1, n2, PDB = params(d)
    n2 > p + 3 || throw(ArgumentError("cov only defined for df2 > dim + 3"))
    n = n1 + n2
    B = Matrix(PDB)
    n1*(n - p - 1)*inv((n2 - p)*(n2 - p - 1)*(n2 - p - 3))*(2inv(n2 - p - 1)*B[i,j]*B[k,l] + B[j,l]*B[i,k] + B[i,l]*B[k,j])
end

function var(d::MatrixFDist, i::Integer, j::Integer)
    p = dim(d)
    n1, n2, PDB = params(d)
    n2 > p + 3 || throw(ArgumentError("var only defined for df2 > dim + 3"))
    n = n1 + n2
    B = Matrix(PDB)
    n1*(n - p - 1)*inv((n2 - p)*(n2 - p - 1)*(n2 - p - 3))*((2inv(n2 - p - 1) + 1)*B[i,j]^2 + B[j,j]*B[i,i])
end

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function matrixfdist_logc0(n1::Real, n2::Real, B::AbstractPDMat)
    #  returns the natural log of the normalizing constant for the pdf
    p = dim(B)
    term1 = -logmvbeta(p, n1 / 2, n2 / 2)
    term2 = (n2 / 2) * logdet(B)
    return term1 + term2
end

function logkernel(d::MatrixFDist, Σ::AbstractMatrix)
    p = dim(d)
    n1, n2, B = params(d)
    ((n1 - p - 1) / 2) * logdet(Σ) - ((n1 + n2) / 2) * logdet(pdadd(Σ, B))
end

_logpdf(d::MatrixFDist, Σ::AbstractMatrix) = logkernel(d, Σ) + d.logc0

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

function _rand!(rng::AbstractRNG, d::MatrixFDist, A::AbstractMatrix)
    Ψ = rand(rng, d.W)
    A .= rand(rng, InverseWishart(d.n2, Ψ) )
end

#  -----------------------------------------------------------------------------
#  Transformation
#  -----------------------------------------------------------------------------

inv(d::MatrixFDist) = ( (n1, n2, B) = params(d); MatrixFDist(n2, n1, inv(B)) )

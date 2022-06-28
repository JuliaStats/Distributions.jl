"""
    MatrixBeta(p, n1, n2)
```julia
p::Int    dimension
n1::Real  degrees of freedom (greater than p - 1)
n2::Real  degrees of freedom (greater than p - 1)
```
The [matrix beta distribution](https://en.wikipedia.org/wiki/Matrix_variate_beta_distribution)
generalizes the beta distribution to ``p\\times p`` real matrices ``\\mathbf{U}``
for which ``\\mathbf{U}`` and ``\\mathbf{I}_p-\\mathbf{U}`` are both positive definite.
If ``\\mathbf{U}\\sim \\textrm{MB}_p(n_1/2, n_2/2)``, then its probability density function is

```math
f(\\mathbf{U}; n_1,n_2) = \\frac{\\Gamma_p(\\frac{n_1+n_2}{2})}{\\Gamma_p(\\frac{n_1}{2})\\Gamma_p(\\frac{n_2}{2})}
|\\mathbf{U}|^{(n_1-p-1)/2}\\left|\\mathbf{I}_p-\\mathbf{U}\\right|^{(n_2-p-1)/2}.
```

If ``\\mathbf{S}_1\\sim \\textrm{W}_p(n_1,\\mathbf{I}_p)`` and
``\\mathbf{S}_2\\sim \\textrm{W}_p(n_2,\\mathbf{I}_p)``
are independent, and we use ``\\mathcal{L}(\\cdot)`` to denote the lower Cholesky factor, then

```math
\\mathbf{U}=\\mathcal{L}(\\mathbf{S}_1+\\mathbf{S}_2)^{-1}\\mathbf{S}_1\\mathcal{L}(\\mathbf{S}_1+\\mathbf{S}_2)^{-\\rm{T}}
```

has ``\\mathbf{U}\\sim \\textrm{MB}_p(n_1/2, n_2/2)``.
"""
struct MatrixBeta{T <: Real, TW} <: ContinuousMatrixDistribution
    W1::TW
    W2::TW
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function MatrixBeta(p::Int, n1::Real, n2::Real)
    p > 0 || throw(ArgumentError("dim must be positive: got $(p)."))
    logc0 = matrixbeta_logc0(p, n1, n2)
    T = Base.promote_eltype(n1, n2, logc0)
    Ip = ScalMat(p, one(T))
    W1 = Wishart(T(n1), Ip)
    W2 = Wishart(T(n2), Ip)
    MatrixBeta{T, typeof(W1)}(W1, W2, T(logc0))
end

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

 show(io::IO, d::MatrixBeta) = show_multline(io, d, [(:n1, d.W1.df), (:n2, d.W2.df)])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{MatrixBeta{T}}, d::MatrixBeta) where T <: Real
    W1 = convert(Wishart{T}, d.W1)
    W2 = convert(Wishart{T}, d.W2)
    MatrixBeta{T, typeof(W1)}(W1, W2, T(d.logc0))
end
Base.convert(::Type{MatrixBeta{T}}, d::MatrixBeta{T}) where {T<:Real} = d

function convert(::Type{MatrixBeta{T}}, W1::Wishart, W2::Wishart, logc0) where T <: Real
    WW1 = convert(Wishart{T}, W1)
    WW2 = convert(Wishart{T}, W2)
    MatrixBeta{T, typeof(WW1)}(WW1, WW2, T(logc0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

size(d::MatrixBeta) = size(d.W1)

rank(d::MatrixBeta) = size(d, 1)

insupport(d::MatrixBeta, U::AbstractMatrix) = isreal(U) && size(U) == size(d) && isposdef(U) && isposdef(I - U)

params(d::MatrixBeta) = (size(d, 1), d.W1.df, d.W2.df)

mean(d::MatrixBeta) = ((p, n1, n2) = params(d); Matrix((n1 / (n1 + n2)) * I, p, p))

@inline partype(d::MatrixBeta{T}) where {T <: Real} = T

#  Konno (1988 JJSS) Corollary 3.3.i
function cov(d::MatrixBeta, i::Integer, j::Integer, k::Integer, l::Integer)
    p, n1, n2 = params(d)
    n = n1 + n2
    Ω = Matrix{partype(d)}(I, p, p)
    n1*n2*inv(n*(n - 1)*(n + 2))*(-(2/n)*Ω[i,j]*Ω[k,l] + Ω[j,l]*Ω[i,k] + Ω[i,l]*Ω[k,j])
end

function var(d::MatrixBeta, i::Integer, j::Integer)
    p, n1, n2 = params(d)
    n = n1 + n2
    Ω = Matrix{partype(d)}(I, p, p)
    n1*n2*inv(n*(n - 1)*(n + 2))*((1 - (2/n))*Ω[i,j]^2 + Ω[j,j]*Ω[i,i])
end

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function matrixbeta_logc0(p::Int, n1::Real, n2::Real)
    #  returns the natural log of the normalizing constant for the pdf
    return -logmvbeta(p, n1 / 2, n2 / 2)
end

function logkernel(d::MatrixBeta, U::AbstractMatrix)
    p, n1, n2 = params(d)
    ((n1 - p - 1) / 2) * logdet(U) + ((n2 - p - 1) / 2) * logdet(I - U)
end

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

#  Mitra (1970 Sankhyā)

function _rand!(rng::AbstractRNG, d::MatrixBeta, A::AbstractMatrix)
    S1   = PDMat( rand(rng, d.W1) )
    S2   = PDMat( rand(rng, d.W2) )
    S    = S1 + S2
    invL = Matrix( inv(S.chol.L) )
    A .= X_A_Xt(S1, invL)
end

#  -----------------------------------------------------------------------------
#  Test utils
#  -----------------------------------------------------------------------------

function _univariate(d::MatrixBeta)
    check_univariate(d)
    p, n1, n2 = params(d)
    return Beta(n1 / 2, n2 / 2)
end

function _rand_params(::Type{MatrixBeta}, elty, n::Int, p::Int)
    n == p || throw(ArgumentError("dims must be equal for MatrixBeta"))
    n1 = elty( n + 1 + abs(10randn()) )
    n2 = elty( n + 1 + abs(10randn()) )
    return n, n1, n2
end

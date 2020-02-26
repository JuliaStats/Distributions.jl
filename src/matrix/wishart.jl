# Wishart distribution
#
#   following the Wikipedia parameterization
#
"""
    Wishart(ν, S)
```julia
ν::Real           degrees of freedom (greater than p - 1)
S::AbstractPDMat  p x p scale matrix
```
The [Wishart distribution](http://en.wikipedia.org/wiki/Wishart_distribution)
generalizes the gamma distribution to ``p\\times p`` real, positive definite
matrices ``\\mathbf{H}``. If ``\\mathbf{H}\\sim W_p(\\nu,\\mathbf{S})``, then its
probability density function is

```math
f(\\mathbf{H};\\nu,\\mathbf{S}) = \\frac{1}{2^{\\nu p/2} \\left|\\mathbf{S}\\right|^{\\nu/2} \\Gamma_p\\left(\\frac {\\nu}{2}\\right ) }{\\left|\\mathbf{H}\\right|}^{(\\nu-p-1)/2} e^{-(1/2)\\operatorname{tr}(\\mathbf{S}^{-1}\\mathbf{H})}.
```

If ``\\nu`` is an integer, then a random matrix ``\\mathbf{H}`` given by

```math
\\mathbf{H} = \\mathbf{X}\\mathbf{X}^{\\rm{T}}, \\quad\\mathbf{X} \\sim MN_{p,\\nu}(\\mathbf{0}, \\mathbf{S}, \\mathbf{I}_{\\nu})
```

has ``\\mathbf{H}\\sim W_p(\\nu, \\mathbf{S})``. For non-integer degrees of freedom,
Wishart matrices can be generated via the [Bartlett decomposition](https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition).
"""
struct Wishart{T<:Real, ST<:AbstractPDMat} <: ContinuousMatrixDistribution
    df::T     # degree of freedom
    S::ST           # the scale matrix
    c0::T     # the logarithm of normalizing constant in pdf
end

#### Constructors

function Wishart(df::T, S::AbstractPDMat{T}) where T<:Real
    p = dim(S)
    df > p - 1 || error("dpf should be greater than dim - 1.")
    c0 = _wishart_c0(df, S)
    R = Base.promote_eltype(T, c0)
    prom_S = convert(AbstractArray{T}, S)
    Wishart{R, typeof(prom_S)}(R(df), prom_S, R(c0))
end

function Wishart(df::Real, S::AbstractPDMat)
    T = Base.promote_eltype(df, S)
    Wishart(T(df), convert(AbstractArray{T}, S))
end

Wishart(df::Real, S::Matrix) = Wishart(df, PDMat(S))

Wishart(df::Real, S::Cholesky) = Wishart(df, PDMat(S))

function _wishart_c0(df::Real, S::AbstractPDMat)
    h_df = df / 2
    p = dim(S)
    h_df * (logdet(S) + p * typeof(df)(logtwo)) + logmvgamma(p, h_df)
end


#### Properties

insupport(::Type{Wishart}, X::Matrix) = isposdef(X)
insupport(d::Wishart, X::Matrix) = size(X) == size(d) && isposdef(X)

dim(d::Wishart) = dim(d.S)
size(d::Wishart) = (p = dim(d); (p, p))
size(d::Wishart, i) = size(d)[i]
rank(d::Wishart) = dim(d)
params(d::Wishart) = (d.df, d.S, d.c0)
@inline partype(d::Wishart{T}) where {T<:Real} = T

### Conversion
function convert(::Type{Wishart{T}}, d::Wishart) where T<:Real
    P = convert(AbstractArray{T}, d.S)
    Wishart{T, typeof(P)}(T(d.df), P, T(d.c0))
end
function convert(::Type{Wishart{T}}, df, S::AbstractPDMat, c0) where T<:Real
    P = convert(AbstractArray{T}, S)
    Wishart{T, typeof(P)}(T(df), P, T(c0))
end

#### Show

show(io::IO, d::Wishart) = show_multline(io, d, [(:df, d.df), (:S, Matrix(d.S))])


#### Statistics

mean(d::Wishart) = d.df * Matrix(d.S)

function mode(d::Wishart)
    r = d.df - dim(d) - 1.0
    if r > 0.0
        return Matrix(d.S) * r
    else
        error("mode is only defined when df > p + 1")
    end
end

function meanlogdet(d::Wishart)
    p = dim(d)
    df = d.df
    v = logdet(d.S) + p * logtwo
    for i = 1:p
        v += digamma(0.5 * (df - (i - 1)))
    end
    return v
end

function entropy(d::Wishart)
    p = dim(d)
    df = d.df
    d.c0 - 0.5 * (df - p - 1) * meanlogdet(d) + 0.5 * df * p
end

#  Gupta/Nagar (1999) Theorem 3.3.15.i
function cov(d::Wishart, i::Integer, j::Integer, k::Integer, l::Integer)
    S = Matrix(d.S)
    d.df * (S[i, k] * S[j, l] + S[i, l] * S[j, k])
end

function var(d::Wishart, i::Integer, j::Integer)
    S = Matrix(d.S)
    d.df * (S[i, i] * S[j, j] + S[i, j] ^ 2)
end

#### Evaluation

function _logpdf(d::Wishart, X::AbstractMatrix)
    df = d.df
    p = dim(d)
    Xcf = cholesky(X)
    0.5 * ((df - (p + 1)) * logdet(Xcf) - tr(d.S \ X)) - d.c0
end

#### Sampling
function _rand!(rng::AbstractRNG, d::Wishart, A::AbstractMatrix)
    _wishart_genA!(rng, dim(d), d.df, A)
    unwhiten!(d.S, A)
    A .= A * A'
end

function _wishart_genA!(rng::AbstractRNG, p::Int, df::Real, A::AbstractMatrix)
    # Generate the matrix A in the Bartlett decomposition
    #
    #   A is a lower triangular matrix, with
    #
    #       A(i, j) ~ sqrt of Chisq(df - i + 1) when i == j
    #               ~ Normal()                  when i > j
    #
    A .= zero(eltype(A))
    for i = 1:p
        @inbounds A[i,i] = rand(rng, Chi(df - i + 1.0))
    end
    for j in 1:p-1, i in j+1:p
        @inbounds A[i,j] = randn(rng)
    end
end

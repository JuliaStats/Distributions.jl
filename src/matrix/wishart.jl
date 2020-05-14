"""
    Wishart(ν, S)
```julia
ν::Real           degrees of freedom (whole number or a real number greater than p - 1)
S::AbstractPDMat  p x p scale matrix
```
The [Wishart distribution](http://en.wikipedia.org/wiki/Wishart_distribution)
generalizes the gamma distribution to ``p\\times p`` real, positive semidefinite
matrices ``\\mathbf{H}``.

If ``\\nu>p-1``, then ``\\mathbf{H}\\sim \\textrm{W}_p(\\nu, \\mathbf{S})``
has rank ``p`` and its probability density function is

```math
f(\\mathbf{H};\\nu,\\mathbf{S}) = \\frac{1}{2^{\\nu p/2} \\left|\\mathbf{S}\\right|^{\\nu/2} \\Gamma_p\\left(\\frac {\\nu}{2}\\right ) }{\\left|\\mathbf{H}\\right|}^{(\\nu-p-1)/2} e^{-(1/2)\\operatorname{tr}(\\mathbf{S}^{-1}\\mathbf{H})}.
```

If ``\\nu\\leq p-1``, then ``\\mathbf{H}`` is rank ``\\nu`` and it has
a density with respect to a suitably chosen volume element on the space of
positive semidefinite matrices. See [here](https://doi.org/10.1214/aos/1176325375).

For integer ``\\nu``, a random matrix given by

```math
\\mathbf{H} = \\mathbf{X}\\mathbf{X}^{\\rm{T}},
\\quad\\mathbf{X} \\sim \\textrm{MN}_{p,\\nu}(\\mathbf{0}, \\mathbf{S}, \\mathbf{I}_{\\nu})
```

has ``\\mathbf{H}\\sim \\textrm{W}_p(\\nu, \\mathbf{S})``.
For non-integer ``\\nu``, Wishart matrices can be generated via the
[Bartlett decomposition](https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition).
"""
struct Wishart{T<:Real, ST<:AbstractPDMat, R<:Integer} <: ContinuousMatrixDistribution
    df::T          # degree of freedom
    S::ST          # the scale matrix
    logc0::T       # the logarithm of normalizing constant in pdf
    rank::R        # rank of a sample
    singular::Bool # singular of nonsingular wishart?
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function Wishart(df::T, S::AbstractPDMat{T}, warn::Bool = true) where T<:Real
    df > 0 || throw(ArgumentError("df must be positive. got $(df)."))
    p = dim(S)
    rnk = p
    singular = df <= p - 1
    if singular
        isinteger(df) || throw(ArgumentError("singular df must be an integer. got $(df)."))
        rnk = convert(Integer, df)
        warn && @warn("got df <= dim - 1; returning a singular Wishart")
    end
    logc0 = wishart_logc0(df, S, rnk)
    R = Base.promote_eltype(T, logc0)
    prom_S = convert(AbstractArray{T}, S)
    Wishart{R, typeof(prom_S), typeof(rnk)}(R(df), prom_S, R(logc0), rnk, singular)
end

function Wishart(df::Real, S::AbstractPDMat, warn::Bool = true)
    T = Base.promote_eltype(df, S)
    Wishart(T(df), convert(AbstractArray{T}, S), warn)
end

Wishart(df::Real, S::Matrix, warn::Bool = true) = Wishart(df, PDMat(S), warn)
Wishart(df::Real, S::Cholesky, warn::Bool = true) = Wishart(df, PDMat(S), warn)

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

show(io::IO, d::Wishart) = show_multline(io, d, [(:df, d.df), (:S, Matrix(d.S))])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{Wishart{T}}, d::Wishart) where T<:Real
    P = convert(AbstractArray{T}, d.S)
    Wishart{T, typeof(P), typeof(d.rank)}(T(d.df), P, T(d.logc0), d.rank, d.singular)
end
function convert(::Type{Wishart{T}}, df, S::AbstractPDMat, logc0, rnk, singular) where T<:Real
    P = convert(AbstractArray{T}, S)
    Wishart{T, typeof(P), typeof(rnk)}(T(df), P, T(logc0), rnk, singular)
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

insupport(::Type{Wishart}, X::AbstractMatrix) = ispossemdef(X)
function insupport(d::Wishart, X::AbstractMatrix)
    size(X) == size(d) || return false
    if d.singular
        return ispossemdef(X, rank(d))
    else
        return isposdef(X)
    end
end

dim(d::Wishart) = dim(d.S)
size(d::Wishart) = (p = dim(d); (p, p))
rank(d::Wishart) = d.rank
params(d::Wishart) = (d.df, d.S)
@inline partype(d::Wishart{T}) where {T<:Real} = T

mean(d::Wishart) = d.df * Matrix(d.S)

function mode(d::Wishart)
    r = d.df - dim(d) - 1.0
    r > 0.0 || throw(ArgumentError("mode is only defined when df > p + 1"))
    return Matrix(d.S) * r
end

function meanlogdet(d::Wishart)
    d.singular && return -Inf
    p = dim(d)
    df = d.df
    v = logdet(d.S) + p * logtwo
    for i = 1:p
        v += digamma(0.5 * (df - (i - 1)))
    end
    return v
end

function entropy(d::Wishart)
    d.singular && throw(ArgumentError("entropy not defined for singular Wishart."))
    p = dim(d)
    df = d.df
    -d.logc0 - 0.5 * (df - p - 1) * meanlogdet(d) + 0.5 * df * p
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

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function wishart_logc0(df::Real, S::AbstractPDMat, rnk::Integer)
    p = dim(S)
    if df <= p - 1
        return singular_wishart_logc0(p, df, S, rnk)
    else
        return nonsingular_wishart_logc0(p, df, S)
    end
end

function logkernel(d::Wishart, X::AbstractMatrix)
    if d.singular
        return singular_wishart_logkernel(d, X)
    else
        return nonsingular_wishart_logkernel(d, X)
    end
end

#  Singular Wishart pdf: Theorem 6 in Uhlig (1994 AoS)
function singular_wishart_logc0(p::Integer, df::Real, S::AbstractPDMat, rnk::Integer)
    h_df = df / 2
    -h_df * (logdet(S) + p * typeof(df)(logtwo)) - logmvgamma(rnk, h_df) + (rnk*(rnk - p) / 2)*typeof(df)(logπ)
end

function singular_wishart_logkernel(d::Wishart, X::AbstractMatrix)
    df = d.df
    p = dim(d)
    r = rank(d)
    L = eigvals(Hermitian(X), (p - r + 1):p)
    0.5 * ((df - (p + 1)) * sum(log.(L)) - tr(d.S \ X))
end

#  Nonsingular Wishart pdf
function nonsingular_wishart_logc0(p::Integer, df::Real, S::AbstractPDMat)
    h_df = df / 2
    -h_df * (logdet(S) + p * typeof(df)(logtwo)) - logmvgamma(p, h_df)
end

function nonsingular_wishart_logkernel(d::Wishart, X::AbstractMatrix)
    df = d.df
    p = dim(d)
    Xcf = cholesky(X)
    0.5 * ((df - (p + 1)) * logdet(Xcf) - tr(d.S \ X))
end

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

function _rand!(rng::AbstractRNG, d::Wishart, A::AbstractMatrix)
    if d.singular
        A .= zero(eltype(A))
        A[:, 1:rank(d)] = randn(rng, dim(d), rank(d))
    else
        _wishart_genA!(rng, dim(d), d.df, A)
    end
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

#  -----------------------------------------------------------------------------
#  Test utils
#  -----------------------------------------------------------------------------

function _univariate(d::Wishart)
    check_univariate(d)
    df, S = params(d)
    α = df / 2
    β = 2Matrix(S)[1]
    return Gamma(α, β)
end

function _rand_params(::Type{Wishart}, elty, n::Int, p::Int)
    n == p || throw(ArgumentError("dims must be equal for Wishart"))
    ν = elty( n - 1 + abs(10randn()) )
    S = (X = 2rand(elty, n, n) .- 1; X * X')
    return ν, S
end

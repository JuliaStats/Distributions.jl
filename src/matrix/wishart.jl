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

function Wishart(df::T, S::AbstractPDMat{T}) where T<:Real
    df > 0 || throw(ArgumentError("df must be positive. got $(df)."))
    p = size(S, 1)
    singular = df <= p - 1
    if singular
        isinteger(df) || throw(
            ArgumentError("df of a singular Wishart distribution must be an integer (got $df)")
        )
    end
    rnk::Integer = ifelse(singular, df, p)
    logc0 = wishart_logc0(df, S, rnk)
    _df, _logc0 = promote(df, logc0)
    Wishart{typeof(_df), typeof(S), typeof(rnk)}(_df, S, _logc0, rnk, singular)
end

function Wishart(df::Real, S::AbstractPDMat)
    T = Base.promote_eltype(df, S)
    Wishart(T(df), convert(AbstractArray{T}, S))
end

Wishart(df::Real, S::Matrix) = Wishart(df, PDMat(S))
Wishart(df::Real, S::Cholesky) = Wishart(df, PDMat(S))

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

show(io::IO, d::Wishart) = show_multline(io, d, [(:df, d.df), (:S, d.S)])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{Wishart{T}}, d::Wishart) where T<:Real
    P = convert(AbstractArray{T}, d.S)
    Wishart{T, typeof(P), typeof(d.rank)}(T(d.df), P, T(d.logc0), d.rank, d.singular)
end
Base.convert(::Type{Wishart{T}}, d::Wishart{T}) where {T<:Real} = d

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

size(d::Wishart) = size(d.S)

rank(d::Wishart) = d.rank
params(d::Wishart) = (d.df, d.S)
@inline partype(d::Wishart{T}) where {T<:Real} = T

mean(d::Wishart) = d.df * Matrix(d.S)

function mode(d::Wishart)
    r = d.df - size(d, 1) - 1
    r > 0 || throw(ArgumentError("mode is only defined when df > p + 1"))
    return Matrix(d.S) * r
end

function meanlogdet(d::Wishart)
    logdet_S = logdet(d.S)
    p = size(d, 1)
    v = logdet_S + p * oftype(logdet_S, logtwo)
    df = oftype(logdet_S, d.df)
    for i in 0:(p - 1)
        v += digamma((df - i) / 2)
    end
    return d.singular ? oftype(v, -Inf) : v
end

function entropy(d::Wishart)
    d.singular && throw(ArgumentError("entropy not defined for singular Wishart."))
    p = size(d, 1)
    df = d.df
    return -d.logc0 - ((df - p - 1) * meanlogdet(d) - df * p) / 2
end

#  Gupta/Nagar (1999) Theorem 3.3.15.i
function cov(d::Wishart, i::Integer, j::Integer, k::Integer, l::Integer)
    S = d.S
    return d.df * (S[i, k] * S[j, l] + S[i, l] * S[j, k])
end

function var(d::Wishart, i::Integer, j::Integer)
    S = d.S
    return d.df * (S[i, i] * S[j, j] + S[i, j] ^ 2)
end

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function wishart_logc0(df::T, S::AbstractPDMat{T}, rnk::Integer) where {T<:Real}
    p = size(S, 1)
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
function singular_wishart_logc0(p::Integer, df::T, S::AbstractPDMat{T}, rnk::Integer) where {T<:Real}
    logdet_S = logdet(S)
    h_df = oftype(logdet_S, df) / 2
    return -h_df * (logdet_S + p * oftype(logdet_S, logtwo)) - logmvgamma(rnk, h_df) + ((rnk * (rnk - p)) // 2) * oftype(logdet_S, logπ)
end

function singular_wishart_logkernel(d::Wishart, X::AbstractMatrix)
    p = size(d, 1)
    r = rank(d)
    L = eigvals(Hermitian(X), (p - r + 1):p)
    return ((d.df - (p + 1)) * sum(log, L) - tr(d.S \ X)) / 2
end

#  Nonsingular Wishart pdf
function nonsingular_wishart_logc0(p::Integer, df::T, S::AbstractPDMat{T}) where {T<:Real}
    logdet_S = logdet(S)
    h_df = oftype(logdet_S, df) / 2
    return -h_df * (logdet_S + p * oftype(logdet_S, logtwo)) - logmvgamma(p, h_df)
end

function nonsingular_wishart_logkernel(d::Wishart, X::AbstractMatrix)
    return ((d.df - (size(d, 1) + 1)) * logdet(cholesky(X)) - tr(d.S \ X)) / 2
end

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

function _rand!(rng::AbstractRNG, d::Wishart, A::AbstractMatrix)
    if d.singular
        axes2 = axes(A, 2)
        r = rank(d)
        randn!(rng, view(A, :, axes2[1:r]))
        fill!(view(A, :, axes2[(r + 1):end]), zero(eltype(A)))
    else
        _wishart_genA!(rng, A, d.df)
    end
    unwhiten!(d.S, A)
    A .= A * A'
end

function _wishart_genA!(rng::AbstractRNG, A::AbstractMatrix, df::Real)
    # Generate the matrix A in the Bartlett decomposition
    #
    #   A is a lower triangular matrix, with
    #
    #       A(i, j) ~ sqrt of Chisq(df - i + 1) when i == j
    #               ~ Normal()                  when i > j
    #
    T = eltype(A)
    z = zero(T)
    axes1 = axes(A, 1)
    @inbounds for (j, jdx) in enumerate(axes(A, 2)), (i, idx) in enumerate(axes1)
        A[idx, jdx] = if i < j
            z
        elseif i > j
            randn(rng, T)
        else
            rand(rng, Chi(df - i + 1))
        end
    end
    return A
end

#  -----------------------------------------------------------------------------
#  Test utils
#  -----------------------------------------------------------------------------

function _univariate(d::Wishart)
    check_univariate(d)
    df, S = params(d)
    α = df / 2
    β = 2 * first(S)
    return Gamma(α, β)
end

function _rand_params(::Type{Wishart}, elty, n::Int, p::Int)
    n == p || throw(ArgumentError("dims must be equal for Wishart"))
    ν = elty(n - 1 + abs(10 * randn()))
    X = rand(elty, n, n)
    X .= 2 .* X .- 1
    S = X * X'
    return ν, S
end

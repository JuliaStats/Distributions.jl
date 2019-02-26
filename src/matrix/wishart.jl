"""
    Wishart <: ContinuousMatrixDistribution

The *Wishart* distribution.

# Constructors

    Wishart(ν|nu|dof=, Φ|Phi|scale=)

Construct a `Wishart` distribution object with `ν` degrees of freedom, and scale matrix `Φ`.

    Wishart(ν|nu|dof=, mean=)
    Wishart(ν|nu|dof=, mode=)

Construct a `Wishart` distribution object with `ν` degrees of freedom, and matching the relevant `mean` or `mode`.

# Details

The Wishart distribution is a multidimensional generalization of the [`Chisq`](@ref)
distribution, and is characterized by a degree of freedom `ν`, and a base matrix `S`.

It arises as the sampling distribution for the covariance of a zero-mean
[`MvNormal`](@ref) distribution.

# External links

- [Wishart distribution on Wikipedia](http://en.wikipedia.org/wiki/Wishart_distribution)
"""
struct Wishart{T<:Real, ST<:AbstractPDMat} <: ContinuousMatrixDistribution
    ν::T     # degree of freedom
    Φ::ST           # the scale matrix
    c0::T     # the logarithm of normalizing constant in pν
end

#### Constructors

function Wishart(ν::T, S::AbstractPDMat{T}) where T<:Real
    p = dim(S)
    ν > p - 1 || error("dpf should be greater than dim - 1.")
    c0 = _wishart_c0(ν, S)
    R = Base.promote_eltype(T, c0)
    prom_S = convert(AbstractArray{T}, S)
    Wishart{R, typeof(prom_S)}(R(ν), prom_S, R(c0))
end

function Wishart(ν::Real, S::AbstractPDMat)
    T = Base.promote_eltype(ν, S)
    Wishart(T(ν), convert(AbstractArray{T}, S))
end

Wishart(ν::Real, S::Matrix) = Wishart(ν, PDMat(S))

Wishart(ν::Real, S::Cholesky) = Wishart(ν, PDMat(S))

@kwdispatch (::Type{D})(;nu=>ν, dof=>ν, Phi=>Φ, scale=>Φ) where {D<:Wishart} begin
    (ν, Φ) -> D(ν, Φ)
    (ν, mean) -> D(ν, mean ./ ν)
    function (ν, mode)
        p = dim(mode)
        @check_args(Wishart, ν > p+1)
        D(ν, mode ./ (ν-p-1))
    end
end

function _wishart_c0(ν::Real, S::AbstractPDMat)
    h_ν = ν / 2
    p = dim(S)
    h_ν * (logdet(S) + p * typeof(ν)(logtwo)) + logmvgamma(p, h_ν)
end


#### Properties

insupport(::Type{Wishart}, X::Matrix) = isposdef(X)
insupport(d::Wishart, X::Matrix) = size(X) == size(d) && isposdef(X)

dim(d::Wishart) = dim(d.Φ)
size(d::Wishart) = (p = dim(d); (p, p))

size(d::Wishart, i) = size(d)[i]
params(d::Wishart) = (d.ν, d.Φ, d.c0)
@inline partype(d::Wishart{T}) where {T<:Real} = T

### Conversion
function convert(::Type{Wishart{T}}, d::Wishart) where T<:Real
    P = AbstractMatrix{T}(d.Φ)
    Wishart{T, typeof(P)}(T(d.ν), P, T(d.c0))
end
function convert(::Type{Wishart{T}}, ν, S::AbstractPDMat, c0) where T<:Real
    P = AbstractMatrix{T}(S)
    Wishart{T, typeof(P)}(T(ν), P, T(c0))
end

#### Show

show(io::IO, d::Wishart) = show_multline(io, d, [(:ν, d.ν), (:S, Matrix(d.Φ))])


#### Statistics

mean(d::Wishart) = d.ν * Matrix(d.Φ)

function mode(d::Wishart)
    r = d.ν - dim(d) - 1.0
    if r > 0.0
        return Matrix(d.Φ) * r
    else
        error("mode is only defined when ν > p + 1")
    end
end

function meanlogdet(d::Wishart)
    p = dim(d)
    ν = d.ν
    v = logdet(d.Φ) + p * logtwo
    for i = 1:p
        v += digamma(0.5 * (ν - (i - 1)))
    end
    return v
end

function entropy(d::Wishart)
    p = dim(d)
    ν = d.ν
    d.c0 - 0.5 * (ν - p - 1) * meanlogdet(d) + 0.5 * ν * p
end


#### Evaluation

function _logpdf(d::Wishart, X::AbstractMatrix)
    ν = d.ν
    p = dim(d)
    Xcf = cholesky(X)
    0.5 * ((ν - (p + 1)) * logdet(Xcf) - tr(d.Φ \ X)) - d.c0
end

#### Sampling

function _rand!(rng::AbstractRNG, d::Wishart, A::AbstractMatrix)
    _wishart_genA!(rng, dim(d), d.ν, A)
    unwhiten!(d.S, A)
    A .= A * A'
end

function _wishart_genA!(rng::AbstractRNG, p::Int, ν::Real, A::AbstractMatrix)
    # Generate the matrix A in the Bartlett decomposition
    #
    #   A is a lower triangular matrix, with
    #
    #       A(i, j) ~ sqrt of Chisq(ν - i + 1) when i == j
    #               ~ Normal()                  when i > j
    #
    A .= zero(eltype(A))
    for i = 1:p
        @inbounds A[i,i] = rand(rng, Chi(ν - i + 1.0))
    end
    for j in 1:p-1, i in j+1:p
        @inbounds A[i,j] = randn(rng)
    end
end

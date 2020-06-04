"""
    MatrixReshaped(D, n, p)
```julia
D::MultivariateDistribution  base distribution
n::Integer   number of rows
p::Integer   number of columns
```
Reshapes a multivariate distribution into a matrix distribution with n rows and
p columns.

"""
struct MatrixReshaped{S<:ValueSupport,D<:MultivariateDistribution{S}} <:
       MatrixDistribution{S}
    d::D
    num_rows::Int
    num_cols::Int
    function MatrixReshaped(
        d::D,
        n::N,
        p::N,
    ) where {
        D<:MultivariateDistribution{
            S,
        },
    } where {S<:ValueSupport} where {N<:Integer}
        (n > 0 && p > 0) || throw(ArgumentError("n and p should be positive"))
        n * p == length(d) ||
        throw(ArgumentError("Dimensions provided ($n x $p) do not match source distribution of length $(length(d))"))
        return new{S,D}(d, n, p)
    end
end

MatrixReshaped(D::MultivariateDistribution, n::Integer) =
    MatrixReshaped(D, n, n)

show(io::IO, d::MatrixReshaped) =
    show_multline(io, d, [(:num_rows, d.num_rows), (:num_cols, d.num_cols)])


#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

size(d::MatrixReshaped) = (d.num_rows, d.num_cols)

length(d::MatrixReshaped) = length(d.d)

rank(d::MatrixReshaped) = minimum(size(d))

function insupport(d::MatrixReshaped, X::AbstractMatrix)
    return isreal(X) && size(d) == size(X) && insupport(d.d, vec(X))
end

mean(d::MatrixReshaped) = reshape(mean(d.d), size(d))
mode(d::MatrixReshaped) = reshape(mode(d.d), size(d))
cov(d::MatrixReshaped, ::Val{true} = Val(true)) =
    reshape(cov(d.d), prod(size(d)), prod(size(d)))
cov(d::MatrixReshaped, ::Val{false}) =
    ((n, p) = size(d); reshape(cov(d), n, p, n, p))
var(d::MatrixReshaped) = reshape(var(d.d), size(d))

params(d::MatrixReshaped) = (d.d, d.num_rows, d.num_cols)

@inline partype(
    d::MatrixReshaped{S,<:MultivariateDistribution{S}},
) where {S<:Real} = S

_logpdf(d::MatrixReshaped, X::AbstractMatrix) = logpdf(d.d, vec(X))

function _rand!(rng::AbstractRNG, d::MatrixReshaped, Y::AbstractMatrix)
    rand!(rng, d.d, view(Y, :))
    return Y
end

vec(d::MatrixReshaped) = d.d

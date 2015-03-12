# Conjugates for
#
#	Dirichlet - Categorical
#	Dirichlet - Multinomial
#


### Dirichlet - Categorical

complete(G::Type{Categorical}, pri::Dirichlet, p::Vector{Float64}) = Categorical(p)

posterior_canon(pri::Dirichlet, ss::CategoricalStats) = DirichletCanon(pri.alpha + ss.h)

function posterior_canon{T<:Integer}(pri::Dirichlet, G::Type{Categorical}, x::Array{T})
	DirichletCanon(add_categorical_counts!(copy(pri.alpha), x))
end

function posterior_canon{T<:Integer}(pri::Dirichlet, G::Type{Categorical}, x::Array{T}, w::Array{Float64})
	DirichletCanon(add_categorical_counts!(copy(pri.alpha), x, w))
end

### Dirichlet - Multinomial

posterior_canon(pri::Dirichlet, ss::MultinomialStats) = DirichletCanon(pri.alpha + ss.scnts)

function posterior_canon{T<:Real}(pri::Dirichlet, G::Type{Multinomial}, x::Matrix{T})
	d = length(pri)
	size(x,1) == d || throw(ArgumentError("Inconsistent argument dimensions."))
	a = add!(sum(x, 2), pri.alpha)
	DirichletCanon(vec(a))
end

function posterior_canon{T<:Real}(pri::Dirichlet, G::Type{Multinomial}, x::Matrix{T}, w::Array{Float64})
	d = length(pri)
	size(x) == (d, length(w)) || throw(ArgumentError("Inconsistent argument dimensions."))
	a = copy(pri.alpha)
	@compat Base.LinAlg.BLAS.gemv!('N', 1.0, map(Float64, x), vec(w), 1.0, a)
	DirichletCanon(a)
end


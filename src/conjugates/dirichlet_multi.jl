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
	d = dim(pri)
	size(x,1) == d || throw(ArgumentError("Inconsistent argument dimensions."))
	a = Array(Float64, d)
	add!(sum!(a, x, 2), pri.alpha)
	DirichletCanon(a)
end

function posterior_canon{T<:Real}(pri::Dirichlet, G::Type{Multinomial}, x::Matrix{T}, w::Array{Float64})
	d = dim(pri)
	if !(size(x,1) == d && size(x,2) == length(w))
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	a = copy(pri.alpha)
	Base.LinAlg.BLAS.gemv!('N', 1.0, float64(x), vec(w), 1.0, a)
	DirichletCanon(a)
end


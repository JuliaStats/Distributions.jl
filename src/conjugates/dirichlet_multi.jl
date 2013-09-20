# Conjugates for
#
#	Dirichlet - Categorical
#	Dirichlet - Multinomial
#

function dirichlet_posupdate!{T<:Integer}(α::Vector{Float64}, ::Type{Categorical}, x::Array{T})
	for i in 1:length(x)
		@inbounds xi = x[i]
		α[xi] += 1.0   # cannot put @inbounds here, as no guarantee xi is inbound
	end
	return α
end

function dirichlet_posupdate!{T<:Integer}(α::Vector{Float64}, ::Type{Categorical}, x::Array{T}, w::Array{Float64})
	if length(x) != length(w)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	for i in 1:length(x)
		@inbounds xi = x[i]
		@inbounds wi = w[i]
		α[xi] += wi
	end
	return α
end

# each column are counts for one experiment
function dirichlet_posupdate!{T<:Real}(α::Vector{Float64}, ::Type{Multinomial}, X::Matrix{T})
	d::Int = length(α)
	if d != size(X, 1)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	n = size(X, 2)
	o = 0
	for j = 1:n
		for i = 1:d
			@inbounds α[i] += X[o+i]
		end
		o += d
	end
	return α
end

function dirichlet_posupdate!{T<:Real}(α::Vector{Float64}, ::Type{Multinomial}, X::Matrix{T}, w::Array{Float64})
	d::Int = length(α)
	if d != size(X, 1)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	n = size(X, 2)
	if n != length(w)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	o = 0
	for j = 1:n
		@inbounds wj = w[j]
		for i = 1:d
			@inbounds α[i] += X[o+i] * wj
		end
		o += d
	end
	return α
end

function posterior{T<:Integer}(prior::Dirichlet, ::Type{Categorical}, x::Array{T})
	Dirichlet(dirichlet_posupdate!(copy(prior.alpha), Categorical, x))
end

function posterior{T<:Integer}(prior::Dirichlet, ::Type{Categorical}, x::Array{T}, w::Array{Float64})
	Dirichlet(dirichlet_posupdate!(copy(prior.alpha), Categorical, x, w))
end

function posterior_mode{T<:Integer}(prior::Dirichlet, ::Type{Categorical}, x::Array{T})
	α = dirichlet_posupdate!(copy(prior.alpha), Categorical, x)
	dirichlet_mode!(α, α, sum(α))
end

function posterior_mode{T<:Integer}(prior::Dirichlet, ::Type{Categorical}, x::Array{T}, w::Array{Float64})
	α = dirichlet_posupdate!(copy(prior.alpha), Categorical, x, w)
	dirichlet_mode!(α, α, sum(α))
end

function posterior{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, x::Vector{T})
	return Dirichlet(prior.alpha + x)
end

function posterior{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T})
	Dirichlet(dirichlet_posupdate!(copy(prior.alpha), Multinomial, X))
end

function posterior{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T}, w::Array{Float64})
	Dirichlet(dirichlet_posupdate!(copy(prior.alpha), Multinomial, X, w))
end

function posterior_mode{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, x::Vector{T})
	α = prior.alpha + x
	dirichlet_mode!(α, α, sum(α))
end

function posterior_mode{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T})
	α = dirichlet_posupdate!(copy(prior.alpha), Multinomial, X)
	dirichlet_mode!(α, α, sum(α))
end

function posterior_mode{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T}, w::Array{Float64})
	α = dirichlet_posupdate!(copy(prior.alpha), Multinomial, X, w)
	dirichlet_mode!(α, α, sum(α))
end


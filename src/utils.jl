# Utility functions

type NoArgCheck end

function allfinite{T<:Real}(x::Array{T})
	for i = 1 : length(x)
		if !(isfinite(x[i]))
			return false
		end
	end
	return true
end

function allzeros{T<:Real}(x::Array{T})
	for i = 1 : length(x)
		if !(x == zero(T))
			return false
		end
	end
	return true
end

function isprobvec(p::Vector{Float64})
    s = 0.
    for i = 1:length(p)
        pi = p[i]
        s += pi
        if pi < 0
            return false
        end
    end      
    return abs(s - 1.0) <= 1.0e-12
end

function pnormalize!{T<:FloatingPoint}(v::AbstractVector{T})
	s = 0.
	n = length(v)
	for i = 1:n
		@inbounds s += v[i]
	end
	for i = 1:n
		@inbounds v[i] /= s
	end
	v
end

function add!(x::AbstractArray, y::AbstractArray)
	n = length(x)
	length(y) == n || throw(DimensionMismatch("Inconsistent array lengths."))
	for i = 1:n
		x[i] += y[i]
	end
	x
end

function multiply!(x::AbstractArray, c::Number)
	for i = 1:length(x)
		@inbounds x[i] *= c
	end
	x
end


# macros for generating functions for support handling
#
# Both lb & ub should be compile-time constants
# otherwise, one should manually specify the methods
#

macro continuous_distr_support(D, lb, ub)
	if isfinite(eval(lb)) && isfinite(eval(ub))  # [lb, ub]
		esc(quote
			isupperbounded(::Union($D, Type{$D})) = true
			islowerbounded(::Union($D, Type{$D})) = true
			isbounded(::Union($D, Type{$D})) = true
			minimum(::Union($D, Type{$D})) = $lb
			maximum(::Union($D, Type{$D})) = $ub
			insupport(::Union($D, Type{$D}), x::Real) = ($lb <= x <= $ub)
		end)

	elseif isfinite(eval(lb))  # [lb, inf)
		esc(quote
			isupperbounded(::Union($D, Type{$D})) = false
			islowerbounded(::Union($D, Type{$D})) = true
			isbounded(::Union($D, Type{$D})) = false
			minimum(::Union($D, Type{$D})) = $lb
			maximum(::Union($D, Type{$D})) = $ub
			insupport(::Union($D, Type{$D}), x::Real) = (isfinite(x) && x >= $lb)
		end)

	elseif isfinite(eval(ub))  # (-inf, ub]
		esc(quote
			isupperbounded(::Union($D, Type{$D})) = true
			islowerbounded(::Union($D, Type{$D})) = false
			isbounded(::Union($D, Type{$D})) = false
			minimum(::Union($D, Type{$D})) = $lb
			maximum(::Union($D, Type{$D})) = $ub
			insupport(::Union($D, Type{$D}), x::Real) = (isfinite(x) && x <= $ub)
		end)

	else   # (-inf, inf)
		esc(quote
			isupperbounded(::Union($D, Type{$D})) = false
			islowerbounded(::Union($D, Type{$D})) = false
			isbounded(::Union($D, Type{$D})) = false
			minimum(::Union($D, Type{$D})) = $lb
			maximum(::Union($D, Type{$D})) = $ub
			insupport(::Union($D, Type{$D}), x::Real) = isfinite(x)
		end)

	end
end

# for checking the input range of quantile functions
# comparison with NaN is always false, so no explicit check is required
macro checkquantile(p,ex)
    :(zero($p) <= $p <= one($p) ? $ex : NaN)
end
macro checkinvlogcdf(lp,ex)
    :($lp <= zero($lp) ? $ex : NaN)
end

# get the variate form (uni/multi/matrix-variate) of a Distribution
variate_form{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VF
variate_form{T<:Distribution}(::Type{T}) = variate_form(super(T))

# get the value support (discrete/continuous) of a Distribution
value_support{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VS
value_support{T<:Distribution}(::Type{T}) = value_support(super(T))

struct PiecewiseLinearDist{T<:Real} <: ContinuousUnivariateDistribution
    x::Vector{T}
    Fx::Vector{Float64}

    function PiecewiseLinearDist(x::AbstractVector{T}, Fx::AbstractVector{<:Real}) where {T<:Real}
        length(x)==length(Fx) || throw(ArgumentError("`x` and `Fx` must have the same length."))
        allunique(x) || throw(ArgumentError("`x` cannot have duplicate values."))
        issorted(x) || throw(ArgumentError("`x` must be sorted."))
        issorted(Fx) || throw(ArgumentError("`Fx` must be sorted."))

        # We enforce that the first y value must be 0 and the last 1
        Fx[1]==0. || throw(ArgumentError("The first value for `Fx` must be 0."))
        Fx[end]==1. || throw(ArgumentError("The last value for `Fx` must be 1."))

        return new{T}(x, Fx)
    end
end

function logpdf(d::PiecewiseLinearDist{T}, x::Real) where {T<:Real}
    insupport(d, x) || return log(zero(T))

    if x==maximum(d)
        return log(zero(T))
    end

    for i=1:length(d.x)
        if x < d.x[i+1]
            # Compute the slope and intercept of the linear part of the cdf in the segment that we're in
            slope = (d.Fx[i+1]-d.Fx[i])/(d.x[i+1]-d.x[i])

            return log(slope)
        end
    end

    # This should never happen, with the checks at construction, we should always
    # exit this function via a return statement in the for loop above.
    error("Internal error.")
end

function cdf(d::PiecewiseLinearDist{T}, x::Real) where {T<:Real}
    if x>maximum(d)
        return one(T)
    end
    
    for i=length(d.x):-1:1
        if x == d.x[i]
            return d.Fx[i]
        elseif x > d.x[i]
            # Compute the slope and intercept of the linear part of the cdf in the segment that we're in
            slope = (d.Fx[i+1]-d.Fx[i])/(d.x[i+1]-d.x[i])
            intercept = d.Fx[i] - slope * d.x[i]

            return intercept + slope * x
        end
    end

    return zero(T)
end

function Statistics.quantile(d::PiecewiseLinearDist, q::Real)
    0 <= q <= 1. || throw(ArgumentError("q must be between 0 and 1."))

    for i=1:length(d.Fx)
        if d.Fx[i]==q
            # We happen to ask for a quantile that sits exactly on a slope-change point
            return d.x[i]
        elseif d.Fx[i] < q < d.Fx[i+1]
            # Compute the slope and intercept of the linear part of the cdf in the segment that we're in
            slope = (d.Fx[i+1]-d.Fx[i])/(d.x[i+1]-d.x[i])
            intercept = d.Fx[i] - slope * d.x[i]

            # Invert the linear function and evaluate it at q
            result = (q-intercept)/slope

            return result
        end
    end

    # This should never happen, with the checks at construction, we should always
    # exit this function via a return statement in the for loop above.
    error("Internal error.")
end

minimum(d::PiecewiseLinearDist) = d.x[1]
maximum(d::PiecewiseLinearDist) = d.x[end]

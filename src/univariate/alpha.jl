immutable Alpha <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Alpha(l::Real, s::Real)
        if s < 0.0
        	error("scale must be non-negative")
        else
        	new(float64(l), float64(s))
        end
    end
end

Alpha(location::Real) = Alpha(location, 1.0)
Alpha() = Alpha(0.0, 1.0)

const Levy = Alpha

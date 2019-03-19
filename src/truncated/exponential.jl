#####
##### Truncated exponential distribition
#####

function mean(d::Truncated{<:Exponential,Continuous})
    θ = d.untruncated.θ
    l, r = extrema(d)           # l is always finite
    if isfinite(r)
        if isapprox(l, r, rtol = √eps() * θ, atol = iszero(l) ? √eps() : 0.0)
            middle(l, r)
        else
            Δ = r - l
            R = -Δ/θ
            θ + middle(l, r) + Δ/2*(exp(R)+1)/expm1(R)
        end
    else
        θ + l
    end
end

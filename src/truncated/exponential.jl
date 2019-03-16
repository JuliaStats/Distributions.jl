#####
##### Truncated exponential distribition
#####

function mean(d::Truncated{<:Exponential,Continuous})
    θ = d.untruncated.θ
    I(y) = y < Inf ? (y + θ)*exp(-y/θ) : zero(y) # integral expression
    (I(minimum(d)) - I(maximum(d))) / (d.ucdf - d.lcdf)
end

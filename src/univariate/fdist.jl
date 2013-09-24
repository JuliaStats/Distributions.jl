immutable FDist <: ContinuousUnivariateDistribution
    ndf::Float64
    ddf::Float64
    function FDist(d1::Real, d2::Real)
    	d1 > zero(d1) && d2 > zero(d2) ||
    	    error("Numerator and denominator degrees of freedom must be positive")
  	new(float64(d1), float64(d2))
    end
end

insupport(::FDist, x::Real) = zero(x) <= x < Inf
insupport(::Type{FDist}, x::Real) = zero(x) <= x < Inf

mean(d::FDist) = 2.0 < d.ddf ? d.ddf / (d.ddf - 2.0) : NaN

median(d::FDist) = quantile(d, 0.5)


mode(d::FDist) = d.ndf <= 2. ? 0.0 : (d.ndf - 2.) / d.ndf * d.ddf / (d.ddf + 2.)
modes(d::FDist) = [mode(d)]

var(d::FDist) = d.ddf > 4. ?  2.0 * d.ddf^2 *
		       (d.ndf + d.ddf - 2.0) /
		       (d.ndf * (d.ddf - 2.0)^2 * (d.ddf - 4.0)) : NaN

skewness(d::FDist) = d.ddf > 6. ?  (2.d.ndf + d.ddf - 2.) * sqrt(8.(d.ddf - 4.)) /
           ((d.ddf - 6.) * sqrt(d.ndf * (d.ndf + d.ddf - 2.))) : NaN

function kurtosis(d::FDist)
    d.ddf <= 8. && return NaN
    a = d.ndf * (5. * d.ddf - 22.) * (d.ndf + d.ddf - 2.) +
        (d.ddf - 4.) * (d.ddf - 2.)^2
    b = d.ndf * (d.ddf - 6.) * (d.ddf - 8.) * (d.ndf + d.ddf - 2.)
    12. * a / b
end

entropy(d::FDist) = (log(d.ddf) -log(d.ndf) 
                     +lgamma(0.5*d.ndf) +lgamma(0.5*d.ddf) -lgamma(0.5*(d.ndf+d.ddf)) 
                     +(1.0-0.5*d.ndf)*digamma(0.5*d.ndf) +(-1.0-0.5*d.ddf)*digamma(0.5*d.ddf)
                     +0.5*(d.ndf+d.ddf)*digamma(0.5*(d.ndf+d.ddf)))


function pdf(d::FDist,x::Real)
    if !insupport(d,x)
        return 0.0
    end
    a = 0.5*d.ndf
    b = 0.5*d.ddf
    u = d.ndf*x
    v = d.ddf
    w = u+v
    brcomp(a,b,u/w,v/w)/x
end

function cdf(d::FDist, x::Real)
    if !insupport(d,x)
        return 0.0
    end
    u = x*d.ndf/d.ddf
    y = u/(one(u)+u)
    cdf(Beta(0.5*d.ndf,0.5*d.ddf),y)
end
function ccdf(d::FDist, x::Real)
    if !insupport(d,x)
        return 0.0
    end
    u = x*d.ndf/d.ddf
    y = u/(one(u)+u)
    ccdf(Beta(0.5*d.ndf,0.5*d.ddf),y)
end

function quantile(d::FDist, p::Real)
    y = quantile(Beta(0.5*d.ndf,0.5*d.ddf),p)
    (d.ddf*y)/(d.ndf*(one(y)-y))
end


rand(d::FDist) = (d.ddf*rand(Chisq(d.ndf)))/(d.ndf*rand(Chisq(d.ddf)))


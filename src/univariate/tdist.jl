immutable TDist <: ContinuousUnivariateDistribution
    df::Float64 # non-integer degrees of freedom allowed
    function TDist(d::Real)
    	d > zero(d) || error("df must be positive")
        new(float64(d))
    end
end

insupport(::TDist, x::Real) = isfinite(x)
insupport(::Type{TDist}, x::Real) = isfinite(x)

mean(d::TDist) = d.df > 1.0 ? 0.0 : NaN

median(d::TDist) = 0.0

mode(d::TDist) = 0.0
modes(d::TDist) = [0.0]

var(d::TDist) = d.df > 2.0 ? d.df / (d.df - 2.0) : d.df > 1.0 ? Inf : NaN

skewness(d::TDist) = d.df > 3.0 ? 3.0 : NaN
kurtosis(d::TDist) = d.df > 4.0 ? 6.0/(d.df-4.0) : NaN

function entropy(d::TDist)
    hdf = 0.5*d.df
    hdfph = hdf + 0.5
    hdfph*(digamma(hdfph) - digamma(hdf)) +
        0.5*log(d.df) + lbeta(hdf,0.5)
end

function pdf(d::TDist, x::Real)
    1.0 / (sqrt(d.df) * beta(0.5, 0.5 * d.df)) *
        (1.0 + x^2 / d.df)^(-0.5 * (d.df + 1.0))
end

# TODO: R claims to do a normal approximation in the tails.
function cdf(d::TDist, x::Real)    
    u = x*x/d.df
    v = 1.0/(1.0+u)
    y = 0.5*bratio(0.5*d.df,0.5,v)
    x <= 0.0 ? y : 1.0-y
end
function ccdf(d::TDist, x::Real)    
    u = x*x/d.df
    v = 1.0/(1.0+u)
    y = 0.5*bratio(0.5*d.df,0.5,v)
    x <= 0.0 ? 1.0-y : y
end

# Based on:
#   G.W. Hill (1970)
#   Algorithm 396: Student's t-quantiles
#   Communications of the ACM,  13 (10): 619-620 
# and subsequent remarks.
function quantile(d::TDist, p::Real)
    if p <= 0.0
        return p == 0.0 ? -Inf : NaN
    elseif p >= 1.0
        return p == 1.0 ? Inf : NaN
    end
    n = d.df
    if n==1
        return quantile(Cauchy(),p)
    elseif n==2
        return sqrt(2.0/(p*(2.0-p))-2.0)
    elseif n < 1
        # throw error?
        return NaN
    end
        
    a = 1.0/(n-0.5)
    b = 48.0 / a^2
    c = ((20700.0*a/b - 98.0)*a - 16.0)*a - 96.36
    dd = ((94.5/(b+c) - 3.0)/b+1.0)*sqrt(halfπ*a)*n
    x = 2.0*dd*p # Hill (1970) gives 2-tail quantile, so need to double p
    y = x^(2.0/n)
    if y > 0.05 + a
        x = Φinv(p)
        y = x^2
        if n < 5
            c += 0.3*(n-4.5)*(x+0.6)
        end
        c = (((0.05*dd*x-5.0)*x-7.0)*x-2.0)*x+b+c
        y = (((((0.4*y+6.3)*y+36.0)*y+94.5)/c-y-3.0)/b+1.0)*x
        y = a*y^2
        y = expm1(y) # use special function (remark by Lozy, 1979)
    else
        y = ((1.0/(((n+6.0)/(n*y)-0.089*dd-0.822)*(n+2.0)*3.0)+0.5/(n+4.0))*y-1.0)*(n+1.0)/(n+2.0)+1.0/y
    end
    q = p < 0.5 ? -sqrt(n*y) : sqrt(n*y)
    # Taylor iterations (remark by Hill, 1981)
    # TODO: tune convergence criteria 
    # e.g. quantile(TDist(28),0.445)
    for i = 1:20
        z = (p - cdf(d,q)) / pdf(d,q)
        delta = (n+1.0)*q*z*z*0.5/(q*q+n) + z
        q += delta
        abs(delta) < 100*eps(q) && break
    end
    q
end

cquantile(d::TDist, p::Real) = -quantile(d,p)

function rand(d::TDist)
    z = randn()
    u = rand(Chisq(d.df))
    return z/sqrt(u/d.df)
end

function cf(d::TDist, t::Real) 
    u = sqrt(d.df*abs(t))
    hdf = 0.5*d.df
    2.0*besselk(hdf,u)*(0.5*u)^hdf/gamma(hdf)
end


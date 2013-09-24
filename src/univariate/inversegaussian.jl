immutable InverseGaussian <: ContinuousUnivariateDistribution
    mu::Float64   # mean
    lambda::Float64 # shape
    function InverseGaussian(mu::Real, lambda::Real)
        mu > zero(mu) || error("mu must be positive")
        lambda > zero(lambda) || error("lambda must be positive")
        new(float64(mu), float64(lambda))
    end
end

InverseGaussian() = InverseGaussian(1.0, 1.0)

@continuous_distr_support InverseGaussian 0.0 Inf

mean(d::InverseGaussian) = d.mu
mode(d::InverseGaussian) = (r=d.mu/d.lambda; d.mu*(sqrt(1.0+2.25*r*r)-1.5*r))
var(d::InverseGaussian) = d.mu*d.mu*d.mu/d.lambda
skewness(d::InverseGaussian) = 3.0*sqrt(d.mu/d.lambda)
kurtosis(d::InverseGaussian) = 15.0*d.mu/d.lambda


function pdf(d::InverseGaussian, x::Real)
    if x <= 0.0
        return 0.0
    end
    dd = x-d.mu
    sqrt(d.lambda/(twoπ*x*x*x))*exp(-d.lambda*dd*dd/(2.0*d.mu*d.mu*x))
end
function logpdf(d::InverseGaussian, x::Real)
    dd = x-d.mu
    0.5*log(d.lambda/(twoπ*x*x*x))-d.lambda*dd*dd/(2.0*d.mu*d.mu*x)
end


function cdf(d::InverseGaussian, x::Real)
    u = sqrt(d.lambda/x)
    v = x/d.mu
    Φ(u*(v-1.0)) + exp(2.0*d.lambda/d.mu)*Φ(-u*(v+1.0))
end
function ccdf(d::InverseGaussian, x::Real)
    u = sqrt(d.lambda/x)
    v = x/d.mu
    Φc(u*(v-1.0)) - exp(2.0*d.lambda/d.mu)*Φ(-u*(v+1.0))
end
function logcdf(d::InverseGaussian, x::Real)
    u = sqrt(d.lambda/x)
    v = x/d.mu
    a = logΦ(u*(v-1.0)) 
    b = 2.0*d.lambda/d.mu + logΦ(-u*(v+1.0))
    a + log1pexp(b-a)
end
function logccdf(d::InverseGaussian, x::Real)
    u = sqrt(d.lambda/x)
    v = x/d.mu
    a = logΦc(u*(v-1.0)) 
    b = 2.0*d.lambda/d.mu + logΦ(-u*(v+1.0))
    a + log1mexp(b-a)
end

function quantile(d::InverseGaussian, p::Real)
    if p <= 0.0 || p >= 1.0
        if p == 1.0
            return inf(Float64)
        elseif p == 0.0
            return 0.0
        else
            return nan(Float64)
        end
    end
    # Whitmore and Yalovsky (1978) approximation
    w = Φinv(p)
    phi = d.lambda/d.mu
    x = d.mu*exp(w/sqrt(phi) - 0.5/phi)
    # one multiplicitive Newton iteration
    xn = x*exp((p-cdf(d,x))/(pdf(d,x)*exp(x)))
    # additive Newton iterations
    while !isapprox(xn,x)
        x = xn
        xn = x + (p-cdf(d,x))/pdf(d,x)
    end
    xn
end


# rand method from:
#   John R. Michael, William R. Schucany and Roy W. Haas (1976) 
#   Generating Random Variates Using Transformations with Multiple Roots
#   The American Statistician , Vol. 30, No. 2, pp. 88-90
function rand(d::InverseGaussian)
    z = randn()
    v = z*z
    w = d.mu*v
    x1 = d.mu + d.mu/(2.0*d.lambda)*(w - sqrt(w*(4.0*d.lambda + w)))
    p1 = d.mu / (d.mu + x1)
    u = rand()
    u >= p1 ? d.mu*d.mu/x1 : x1
end

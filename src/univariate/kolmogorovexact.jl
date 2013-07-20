# Distribution of the (two-sided) Kolmogorov-Smirnoff statistic
#   D_n = \sup_x |\hat{F}_n(x) -F(x)|
#   sqrt(n) D_n converges a.s. to the Kolmogorov distribution.
immutable KSDist <: ContinuousUnivariateDistribution
    n::Int
end

insupport(d::KSDist, x::Real) = 1/(2*d.n) <= x <= 1.0

# Simard and L'Ecuyer (2011) meta-algorithm
function cdf(d::KSDist,x::Real)
    n = d.n
    b = x*n
    # known exact values
    if b <= 0.5
        return 0.0
    elseif b <= 1.0
        # could be improved
        return exp(lfact(n)+n*(log(2.0*b-1.0)-log(n)))
    elseif x >= 1.0
        return 1.0
    elseif b >= n-1
        return 1.0 - 2.0*(1.0-x)^n
    end

    a = b*x
    if a >= 18.0
        return 1.0
    elseif n <= 140
        if a <= 0.754693
            return durbin(d,x)
        elseif a <= 4.0
            return pomeranz(d,x)
        else
            # Miller (1956) approximation
            return 1.0 - 2.0*ccdf(KSOneSided(d.n),x)
        end
    elseif n <= 10_000
        if b*sqrt(n) <= 1.4
            return durbin(d,x)
        end
    end
    pelzgood(d,x)
end

# Simard and L'Ecuyer (2011) meta-algorithm
function ccdf(d::KSDist,x::Real)    
    n = d.n
    b = x*n
    # Ruben and Gambino (1982) known exact values
    if b <= 0.5
        return 1.0
    elseif b <= 1.0
        return 1.0-exp(lfact(n)+n*(log(2.0*b-1.0)-log(n)))
    elseif x >= 1.0
        return 0.0
    elseif b >= n-1
        return 2.0*(1.0-x)^n
    end

    a = b*x
    if a >= 370.0
        return 0.0
    elseif a >= 4.0 || (n > 140 && a >= 2.2)
        # Miller (1956) approximation
        return 2.0*ccdf(KSOneSided(d.n),x)
    end
    1.0-cdf(d,x)
end


# Durbin matrix CDF method, based on Marsaglia, Tsang and Wang (2003)
# modified to avoid need for exponent tracking
function durbin(d::KSDist,x::Float64)
    # a = x*x*n
    # if (a > 7.24) || (a > 3.76) && (n > 99))
    #    return 1 - 2.0*exp(-(2.000071+.331/sqrt(n)+1.409/n)*a)
    # end

    n = d.n
    k = iceil(n*x) 
    h = k-n*x

    m = 2*k-1
    H = Array(Float64,m,m)    
    for i = 1:m, j = 1:m
        H[i,j] = i-j+1 >= 0 ? 1.0 : 0.0        
    end
    for i = 1:m
        H[i,1] -= h^i
        H[m,i] -= h^(m-i+1)
    end
    H[m,1] += h > 0.5 ? (2*h-1) : 0.0
    for i = 1:m, j = 1:m
        # we can avoid keeping track of the exponent by dividing by e
        # (from Stirling's approximation)
        H[i,j] /= e
        for g = 1:max(i-j+1,0)
            H[i,j] /= g
        end
    end

    Q = H^n
    s = Q[k,k]
    # need to multiply by n!*(e/n)^n
    if n < 500
        for i = 1:n
            s *= i/n*e 
        end
    else
        # 3rd-order Stirling's approximation more accurate for large n
        twn = 12.0*n
        s*sqrt(2.0*pi*n)*(1.0 + twn\(1 + (2.0*twn)\(1 - (15.0*twn)\139.0)))
    end
    s
end

# CDF method of Pelz and Good (1976)
# based on asymptotic series, accurate for large n
function pelzgood(d,x)
    n = d.n
    sqrtn = sqrt(n)
    z = x/sqrtn

    K0 = cdf(Kolmogorov(),z)

    K1 = x

    K0 + K1/sqrtn + K2/n + K3/(n*sqrtn)
end    

function pomeranz(d,x)
    n = d.n
    t = n*x


    t = big(x)*n

end
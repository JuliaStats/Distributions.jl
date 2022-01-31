# Edgeworth expansion approximations of sums, means and z statistics of iid variables.
# Quantiles are computed via the Cornish-Fisher expansion
# TODO: make the expansion of arbitrary order, using cumulants and Hermite polynomials

# Edgeworth approximation of the Z statistic
# EdgeworthSum and EdgeworthMean are both defined in terms of this
abstract type EdgeworthAbstract <: ContinuousUnivariateDistribution end

skewness(d::EdgeworthAbstract) = skewness(d.dist) / sqrt(d.n)
kurtosis(d::EdgeworthAbstract) = kurtosis(d.dist) / d.n

struct EdgeworthZ{D<:UnivariateDistribution} <: EdgeworthAbstract
    dist::D
    n::Float64

    function EdgeworthZ{D}(d::UnivariateDistribution, n::Real; check_args::Bool=true) where {D<:UnivariateDistribution}
        @check_args EdgeworthZ (n, n > zero(n))
        new{D}(d, n)
    end
end
function EdgeworthZ(d::UnivariateDistribution, n::Real; check_args::Bool=true)
    return EdgeworthZ{typeof(d)}(d, n; check_args=check_args)
end

mean(d::EdgeworthZ) = 0.0
var(d::EdgeworthZ) = 1.0


function pdf(d::EdgeworthZ, x::Float64)
    s = skewness(d)
    k = kurtosis(d)
    x2 = x*x
    pdf(Normal(0,1),x)*(1 + s*x*(x2-3.0)/6.0 +
                        k*(x2*(x2-6.0)+3.0)/24.0
                        + s*s*(x2*(x2*(x2-15.0)+45.0)-15.0)/72.0)
end
function cdf(d::EdgeworthZ, x::Float64)
    s = skewness(d)
    k = kurtosis(d)
    x2 = x*x
    cdf(Normal(0,1),x) -
    pdf(Normal(0,1),x)*(
                        s*(x2-1.0)/6.0 +
                        k*x*(x2-3.0)/24.0
                        + s*s*x*(x2*(x2-10.0)+15.0)/72.0)
end
function ccdf(d::EdgeworthZ, x::Float64)
    s = skewness(d)
    k = kurtosis(d)
    x2 = x*x
    ccdf(Normal(0,1),x) +
    pdf(Normal(0,1),x)*(
                        s*(x2-1.0)/6.0 +
                        k*x*(x2-3.0)/24.0
                        + s*s*x*(x2*(x2-10.0)+15.0)/72.0)
end


# Cornish-Fisher expansion.
function quantile(d::EdgeworthZ, p::Float64)
    s = skewness(d)
    k = kurtosis(d)
    z = quantile(Normal(0,1),p)
    z2 = z*z
    z + s*(z2-1)/6.0 + k*z*(z2-3)/24.0 - s*s/36.0*z*(2.0*z2-5.0)
end

function cquantile(d::EdgeworthZ, p::Float64)
    s = skewness(d)
    k = kurtosis(d)
    z = cquantile(Normal(0,1),p)
    z2 = z*z
    z + s*(z2-1)/6.0 + k*z*(z2-3)/24.0 - s*s/36.0*z*(2.0*z2-5.0)
end



# Edgeworth approximation of the sum
struct EdgeworthSum{D<:UnivariateDistribution} <: EdgeworthAbstract
    dist::D
    n::Float64
    function EdgeworthSum{D}(d::UnivariateDistribution, n::Real; check_args::Bool=true) where {D<:UnivariateDistribution}
        @check_args EdgeworthSum (n, n > zero(n))
        new{D}(d, n)
    end
end
function EdgeworthSum(d::UnivariateDistribution, n::Real; check_args::Bool=true)
    return EdgeworthSum{typeof(d)}(d, n; check_args=check_args)
end

mean(d::EdgeworthSum) = d.n*mean(d.dist)
var(d::EdgeworthSum) = d.n*var(d.dist)

# Edgeworth approximation of the mean
struct EdgeworthMean{D<:UnivariateDistribution} <: EdgeworthAbstract
    dist::D
    n::Float64
    function EdgeworthMean{D}(d::UnivariateDistribution, n::Real; check_args::Bool=true) where {D<:UnivariateDistribution}
        # although n would usually be an integer, no methods are require this
        @check_args EdgeworthMean (n, n > zero(n), "n must be positive")
        new{D}(d, Float64(n))
    end
end
function EdgeworthMean(d::UnivariateDistribution, n::Real; check_args::Bool=true)
    return EdgeworthMean{typeof(d)}(d, n; check_args=check_args)
end

mean(d::EdgeworthMean) = mean(d.dist)
var(d::EdgeworthMean) = var(d.dist) / d.n

function pdf(d::EdgeworthAbstract, x::Float64)
    m, s = mean(d), std(d)
    pdf(EdgeworthZ(d.dist,d.n),(x-m)/s)/s
end

cdf(d::EdgeworthAbstract, x::Float64) = cdf(EdgeworthZ(d.dist,d.n), (x-mean(d))/std(d))

ccdf(d::EdgeworthAbstract, x::Float64) = ccdf(EdgeworthZ(d.dist,d.n), (x-mean(d))/std(d))

quantile(d::EdgeworthAbstract, p::Float64) = mean(d) + std(d)*quantile(EdgeworthZ(d.dist,d.n), p)
cquantile(d::EdgeworthAbstract, p::Float64) = mean(d) + std(d)*cquantile(EdgeworthZ(d.dist,d.n), p)

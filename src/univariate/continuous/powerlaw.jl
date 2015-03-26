immutable PowerLaw <: ContinuousUnivariateDistribution
    α::Float64
    β::Float64

    function PowerLaw(α::Real, β::Real)
        (α > zero(α) && β > zero(β)) || error("PowerLaw: shape and scale must be positive")
        @compat new(Float64(α), Float64(β))
    end

    PowerLaw(α::Real) = PowerLaw(α, 1.0)
    PowerLaw() = new(2.0, 1.0)
end

@distr_support PowerLaw d.β Inf


#### Parameters

shape(d::PowerLaw) = d.α
scale(d::PowerLaw) = d.β

params(d::PowerLaw) = (d.α, d.β)


#### Statistics

mean(d::PowerLaw) = ((α, β) = params(d); α -= 1; α > 1.0 ? α * β / (α - 1.0) : Inf)
median(d::PowerLaw) = ((α, β) = params(d); α -= 1; β * 2.0 ^ (1.0 / α))
mode(d::PowerLaw) = d.β

function var(d::PowerLaw)
    (α, β) = params(d)
    α -= 1
    α > 2.0 ? (β^2 * α) / ((α - 1.0)^2 * (α - 2.0)) : Inf
end

function skewness(d::PowerLaw)
    α = shape(d)
    α -= 1
    α > 3.0 ? ((2.0 * (1.0 + α)) / (α - 3.0)) * sqrt((α - 2.0) / α) : NaN
end

function kurtosis(d::PowerLaw)
    α = shape(d)
    α -= 1
    α > 4.0 ? (6.0 * (α^3 + α^2 - 6.0 * α - 2.0)) / (α * (α - 3.0) * (α - 4.0)) : NaN
end

entropy(d::PowerLaw) = ((α, β) = params(d); α -= 1; log(β / α) + 1.0 / α + 1.0)


#### Evaluation

function pdf(d::PowerLaw, x::Float64)
    (α, β) = params(d)
    α -= 1
    x >= β ? α * (β / x)^α * (1.0 / x) : 0.0
end

function logpdf(d::PowerLaw, x::Float64)
    (α, β) = params(d)
    α -= 1
    x >= β ? log(α) + α * log(β) - (α + 1.0) * log(x) : -Inf
end

function ccdf(d::PowerLaw, x::Float64)
    (α, β) = params(d)
    α -= 1
    x >= β ? (β / x)^α : 1.0
end

cdf(d::PowerLaw, x::Float64) = 1.0 - ccdf(d, x)

function logccdf(d::PowerLaw, x::Float64)
    (α, β) = params(d)
    α -= 1
    x >= β ? α * log(β / x) : 0.0
end

logcdf(d::PowerLaw, x::Float64) = log1p(-ccdf(d, x))

cquantile(d::PowerLaw, p::Float64) = d.β / p^(1.0 / (d.α - 1.0))
quantile(d::PowerLaw, p::Float64) = cquantile(d, 1.0 - p)


#### Sampling

rand(d::PowerLaw) = d.β * exp(randexp() / (d.α -1.0))

#rand(d::PowerLaw) = d.β / (1.0 - rand())^(1.0 / (d.α - 1.0))


#### Fit model

function fit_mle{T<:Real}(::Type{PowerLaw}, x::Vector{T}, β=find_min(x); return_all::Bool=false)
    x = x[x.>=β]
    n = float(length(x))
    α = 1.0 + n/(sum(log(x)) - n*log(β))
    if return_all
        PowerLaw(α, β), (α - 1.0)/sqrt(n)
    else
        PowerLaw(α, β)
    end
end


#### Find best β by minimize Kolmogorov–Smirnov test


function _ksstats{T<:Real}(x::AbstractVector{T}, d::UnivariateDistribution)
    n = length(x)
    cdfs = cdf(d, sort(x))
    δp = maximum((1:n) / n - cdfs)
    δn = -minimum((0:n-1) / n - cdfs)
    δ = max(δn, δp)
    (n, δ, δp, δn)
end

function ks_distance{T<:Real}(x::Vector{T}, β=minimum(x); return_all::Bool=false)
    if return_all
        _ksstats(x, fit_mle(PowerLaw, x, β))
    else
        _ksstats(x, fit_mle(PowerLaw, x, β))[2]
    end
end

# Find best β by enumerate elements of x between (xmin, xmax), may be take a while
function _find_xmin{T<:Real}(x::Vector{T}, xmin=minimum(x), xmax=maximum(x); return_all::Bool=false)
    D(β) = ks_distance(x, β)
    xmins = x[xmin .<= x .<= xmax]
    xmins = unique(xmins)
    Ds = map(D,xmins)
    Dmin, idx = findmin(Ds)
    if return_all
        xmins, Ds
    else
        xmins[idx], Dmin
    end
end

# Find best β by enumerate elements in xmins
function find_xmin{T<:Real}(x::Vector{T}, xmins::Vector{T}=x; return_all::Bool=false)
    D(β) = ks_distance(x, β)
    Ds = map(D,xmins)
    Dmin, idx = findmin(Ds)
    if return_all
        xmins, Ds
    else
        xmins[idx], Dmin
    end
end

# Find best β effectively, recommend to use
function find_xmin{T<:Real}(x::Vector{T}, xmin=minimum(x), xmax=maximum(x); method::Symbol=:brent)
    D(β) = ks_distance(x, β)
    nbin = 5
    xmins = linspace(xmin,xmax,nbin)
    opts = zeros(nbin-1)
    for i=1:nbin
        opz = Optim.optimize(D, xmins[i], xmins[i+1], method=method)
        opts[i] = opz.minimum
    end
    minimum(opts)
end


#### p-value

# Generate synthetic data sets for p-value calculator
function generate_synthetic_data{T<:Real}(xtail::Vector{T}, n::Int, d::PowerLaw)
    xx = zeros(n)
    ntail = length(xtail)
    p = ntail/n
    if ntail>0
        for i=1:n
            if rand() < p
                xx[i] = xtail[rand(1:ntail)]
            else
                xx[i] = rand(d)
            end
        end
    else
        xx = rand(d,n)
    end
    xx
end

# Calculator p-value for empirical data
# if p-value < 0.1, we can safely rule out the power law hypothesis
# while p-value >= 0.1 is not a sufficient condition for a power law hypothesis
# the higher the p-value, the stronger the power law hypothesis
function pvalue{T<:Real}(x::Vector{T}, N::Int=1000) # N is the number of synthetic sets
    n = length(x)
    β = find_xmin(x)
    d = fit_mle(PowerLaw, x, β)
    D = _ksstats(x, d)[2]
    xtail = x[x.<β]
    cnt = 0
    for i=1:N
        xx = generate_synthetic_data(xtail, n, d)
        if _ksstats(xx, d)[2] > D
            cnt += 1
        end
    end
    cnt/N
end

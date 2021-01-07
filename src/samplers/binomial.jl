
# compute probability vector of a Binomial distribution
function binompvec(n::Int, p::Float64)
    pv = Vector{Float64}(undef, n+1)
    if p == 0.0
        fill!(pv, 0.0)
        pv[1] = 1.0
    elseif p == 1.0
        fill!(pv, 0.0)
        pv[n+1] = 1.0
    else
        q = 1.0 - p
        a = p / q
        @inbounds pv[1] = pk = q ^ n
        for k = 1:n
            @inbounds pv[k+1] = (pk *= ((n - k + 1) / k) * a)
        end
    end
    return pv
end

# Geometric method:
#
#   Devroye. L.
#   "Generating the maximum of independent identically  distributed random variables"
#   Computers & Mathematics with Applications, Volume 6, Issue 3, 1980, Pages 305-315.
#
struct BinomialGeomSampler <: Sampleable{Univariate,Discrete}
    comp::Bool
    n::Int
    scale::Float64
end

BinomialGeomSampler() = BinomialGeomSampler(false, 0, 0.0)

function BinomialGeomSampler(n::Int, prob::Float64)
    if prob <= 0.5
        comp = false
        scale = -1.0/log1p(-prob)
    else
        comp = true
        scale = prob < 1.0 ? -1.0/log(prob) : Inf
    end
    BinomialGeomSampler(comp, n, scale)
end

function rand(rng::AbstractRNG, s::BinomialGeomSampler)
    y = 0
    x = 0
    n = s.n
    while true
        er = randexp(rng)
        v = er * s.scale
        if v > n  # in case when v is very large or infinity
            break
        end
        y += ceil(Int,v)
        if y > n
            break
        end
        x += 1
    end
    (s.comp ? s.n - x : x)::Int
end


# BTPE algorithm from:
#
#   Kachitvichyanukul, V.; Schmeiser, B. W.
#   "Binomial random variate generation."
#   Comm. ACM 31 (1988), no. 2, 216–222.
#
# Note: only use this sampler when n * min(p, 1-p) is large enough
#       e.g., it is greater than 20.
#
struct BinomialTPESampler <: Sampleable{Univariate,Discrete}
    comp::Bool
    n::Int
    r::Float64
    q::Float64
    nrq::Float64
    M::Float64
    Mi::Int
    p1::Float64
    p2::Float64
    p3::Float64
    p4::Float64
    xM::Float64
    xL::Float64
    xR::Float64
    c::Float64
    λL::Float64
    λR::Float64
end

BinomialTPESampler() =
    BinomialTPESampler(false, 0, 0., 0., 0., 0., 0,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)

function BinomialTPESampler(n::Int, prob::Float64)
    if prob <= 0.5
        comp = false
        r = prob
        q = 1.0 - prob
    else
        comp = true
        r = 1.0 - prob
        q = prob
    end

    nrq = n*r*q
    fM = (n+1)*r #
    M = floor(fM)
    Mi = round(Integer, M)
    p1 = floor(2.195*sqrt(nrq)-4.6*q) + 0.5
    xM = M+0.5
    xL = xM-p1
    xR = xM+p1
    c = 0.134 + 20.5/(15.3+M)
    a = (fM-xL)/(fM-xL*r) #
    λL = a*(1.0 + 0.5*a)
    a = (xR-fM)/(xR*q) #
    λR = a*(1.0 + 0.5*a)
    p2 = p1*(1.0 + 2.0*c)
    p3 = p2 + c/λL
    p4 = p3 + c/λR

    BinomialTPESampler(comp,n,r,q,nrq,M,Mi,p1,p2,p3,p4,
                       xM,xL,xR,c,λL,λR)
end

function rand(rng::AbstractRNG, s::BinomialTPESampler)
    y = 0
    while true
        # Step 1
        u = s.p4*rand(rng)
        v = rand(rng)
        if u <= s.p1
            y = floor(Int,s.xM-s.p1*v+u)
            # Goto 6
            break
        elseif u <= s.p2 # Step 2
            x = s.xL + (u-s.p1)/s.c
            v = v*s.c+1.0-abs(s.M-x+0.5)/s.p1
            if v > 1
                # Goto 1
                continue
            end
            y = floor(Int,x)
            # Goto 5
        elseif u <= s.p3 # Step 3
            y = floor(Int,s.xL + log(v)/s.λL)
            if y < 0
                # Goto 1
                continue
            end
            v *= (u-s.p2)*s.λL
            # Goto 5
        else # Step 4
            y = floor(Int,s.xR-log(v)/s.λR)
            if y > s.n
                # Goto 1
                continue
            end
            v *= (u-s.p3)*s.λR
            # Goto 5
        end

        # Step 5
        # 5.0
        k = abs(y-s.Mi)
        if (k <= 20) || (k >= 0.5*s.nrq-1)
            # 5.1
            S = s.r/s.q
            a = S*(s.n+1)
            F = 1.0
            if s.Mi < y
                for i = (s.Mi+1):y
                    F *= a/i-S
                end
            elseif s.Mi > y
                for i = (y+1):s.Mi
                    F /= a/i-S
                end
            end
            if v > F
                # Goto 1
                continue
            end
            # Goto 6
            break
        else
            # 5.2
            ρ = (k/s.nrq)*((k*(k/3.0+0.625)+1.0/6.0)/s.nrq+0.5)
            t = -k^2/(2.0*s.nrq)
            A = log(v)
            if A < t - ρ
                # Goto 6
                break
            elseif A > t + ρ
                # Goto 1
                continue
            end

            # 5.3
            x1 = Float64(y+1)
            f1 = Float64(s.Mi+1)
            z = Float64(s.n+1-s.Mi)
            w = Float64(s.n-y+1)

            if A > (s.xM*log(f1/x1) + ((s.n-s.Mi)+0.5)*log(z/w) + (y-s.Mi)*log(w*s.r/(x1*s.q)) +
                    lstirling_asym(f1) + lstirling_asym(z) + lstirling_asym(x1) + lstirling_asym(w))
                # Goto 1
                continue
            end

            # Goto 6
            break
        end
    end
    # 6
    (s.comp ? s.n - y : y)::Int
end


# Constructing an alias table by directly computing the probability vector
#
struct BinomialAliasSampler <: Sampleable{Univariate,Discrete}
    table::AliasTable
end

BinomialAliasSampler(n::Int, p::Float64) = BinomialAliasSampler(AliasTable(binompvec(n, p)))

rand(rng::AbstractRNG, s::BinomialAliasSampler) = rand(rng, s.table) - 1


# Integrated Polyalgorithm sampler that automatically chooses the proper one
#
# It is important for type-stability
#
mutable struct BinomialPolySampler <: Sampleable{Univariate,Discrete}
    use_btpe::Bool
    geom_sampler::BinomialGeomSampler
    btpe_sampler::BinomialTPESampler
end

function BinomialPolySampler(n::Int, p::Float64)
    q = 1.0 - p
    if n * min(p, q) > 20
        use_btpe = true
        geom_sampler = BinomialGeomSampler()
        btpe_sampler = BinomialTPESampler(n, p)
    else
        use_btpe = false
        geom_sampler = BinomialGeomSampler(n, p)
        btpe_sampler = BinomialTPESampler()
    end
    BinomialPolySampler(use_btpe, geom_sampler, btpe_sampler)
end

BinomialPolySampler(n::Real, p::Real) = BinomialPolySampler(round(Int, n), Float64(p))

rand(rng::AbstractRNG, s::BinomialPolySampler) =
    s.use_btpe ? rand(rng, s.btpe_sampler) : rand(rng, s.geom_sampler)

sampler(d::Binomial) = min(d.prob,1.0-d.prob)*d.size <= 20.0 ? BinomialGeomSampler(d) : BinomialTPESampler(d)

rand(d::Binomial) = rand(sampler(d))
rand!(d::Binomial,a::Array) = rand!(sampler(d),a)

# Geometric method:
#   Devroye. L. 
#   "Generating the maximum of independent identically  distributed random variables" 
#   Computers and Marhemafics with Applicalions 6, 1960, 305-315.
immutable BinomialGeomSampler <: AbstractSampler{Binomial}
    comp::Bool
    n::Int
    scale::Float64
end
function BinomialGeomSampler(d::Binomial)
    if d.prob <= 0.5
        comp = false
        scale = -1.0/log1p(-d.prob)
    else
        comp = true
        scale = -1.0/log(d.prob)
    end
    BinomialGeomSampler(comp,d.size,scale)
end

function rand(s::BinomialGeomSampler)
    y = 0
    x = 0
    while true
        er = Base.Random.randmtzig_exprnd()
        y += iceil(er*s.scale)
        if y > s.n
            break
        end
        x += 1
    end
    s.comp ? s.n - x : x
end


# BTPE algorithm from:
#   Kachitvichyanukul, V.; Schmeiser, B. W. 
#   "Binomial random variate generation." 
#   Comm. ACM 31 (1988), no. 2, 216–222. 
immutable BinomialTPESampler <: AbstractSampler{Binomial}
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

function BinomialTPESampler(d::Binomial)
    n = d.size
    if d.prob <= 0.5
        comp = false
        r = d.prob
        q = 1.0-d.prob
    else
        comp = true
        r = 1.0-d.prob
        q = d.prob
    end

    nrq = n*r*q
    fM = (n+1)*r #
    M = floor(fM)
    Mi = integer(M)
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

function rand(s::BinomialTPESampler)
    y = 0
    while true
        # Step 1
        u = s.p4*rand()
        v = rand()
        if u <= s.p1
            y = ifloor(s.xM-s.p1*v+u)
            # Goto 6
            break
        elseif u <= s.p2 # Step 2
            x = s.xL + (u-s.p1)/s.c
            v = v*s.c+1.0-abs(s.M-x+0.5)/s.p1
            if v > 1
                # Goto 1
                continue
            end
            y = ifloor(x)
            # Goto 5
        elseif u <= s.p3 # Step 3
            y = ifloor(s.xL + log(v)/s.λL)
            if y < 0
                # Goto 1
                continue
            end
            v *= (u-s.p2)*s.λL
            # Goto 5
        else # Step 4
            y = ifloor(s.xR-log(v)/s.λR)
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
            x1 = float64(y+1)
            f1 = float64(s.Mi+1)
            z = float64(s.n+1-s.Mi)
            w = float64(s.n-y+1)

            if A > (s.xM*log(f1/x1) + ((s.n-s.Mi)+0.5)*log(z/w) + (y-s.Mi)*log(w*s.r/(x1*s.q)) +
                    lstirling(f1) + lstirling(z) + lstirling(x1) + lstirling(w))
                # Goto 1
                continue
            end                
            
            # Goto 6
            break
        end
    end
    # 6
    s.comp ? s.n - y : y
end

function lstirling(x)
    ix2 = 1.0/(x*x)
    (13860.0 - (462.0 - (132.0 - (99.0 - 140.0*ix2)*ix2)*ix2)*ix2)/(x*166320.0)
end

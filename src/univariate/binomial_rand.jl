# rand meta-algorithm
# most other methods assume prob <= 0.5
function rand(d::Binomial)
    p, n = d.prob, d.size
    if p <= 0.5
        r = p
    else
        r = 1.0-p
    end
    if r*n <= 10.0
        y = rand_bg(Binomial(n,r))
    else
        y = rand_btpe(Binomial(n,r))
    end
    p <= 0.5 ? y : n-y
end


# simplest algorithm
function rand_bu(d::Binomial)
    p, n = d.prob, d.size
    y = 0
    for i = 1:n
        if rand() <= p
            y += 1
        end
    end
end

# Geometric method:
#   Devroye. L. 
#   "Generating the maximum of independent identically  distributed random variables" 
#   Computers and Marhemafics with Applicalions 6, 1960, 305-315.
function rand_bg(d::Binomial)
    p, n = d.prob, d.size
    y = 0
    x = -1
    while true
        y += rand(Geometric(p)) +1      
        x += 1
        if y > n
            return x
        end
    end
end


# BTPE algorithm from:
#   Kachitvichyanukul, V.; Schmeiser, B. W. 
#   "Binomial random variate generation." 
#   Comm. ACM 31 (1988), no. 2, 216–222. 
function rand_btpe(d::Binomial)
    # Step 0
    r, n = d.prob, d.size
    q = 1.0 - r
    nrq = n*r*q
    fM = (n+1)*r
    M = floor(fM)
    Mi = integer(M)
    p1 = floor(2.195*sqrt(nrq)-4.6*q) + 0.5
    xM = M+0.5
    xL = xM-p1
    xR = xM+p1
    c = 0.134 + 20.5/(15.3+M)
    a = (fM-xL)/(fM-xL*r)
    λL = a*(1.0 + 0.5*a)
    a = (xR-fM)/(xR*q)
    λR = a*(1.0 + 0.5*a)
    p2 = p1*(1.0 + 2.0*c)
    p3 = p2 + c/λL
    p4 = p3 + c/λR

    y = 0

    while true
        # Step 1
        u = p4*rand()
        v = rand()
        if u <= p1
            y = ifloor(xM-p1*v+u)
            # Goto 6
            return y

        elseif u <= p2 # Step 2
            x = xL + (u-p1)/c
            v = v*c+1.0-abs(M-x+0.5)/p1
            if v > 1
                # Goto 1
                continue
            end
            y = ifloor(x)
            # Goto 5
            
        elseif u <= p3 # Step 3
            y = ifloor(xL + log(v)/λL)
            if y < 0
                # Goto 1
                continue
            end
            v *= (u-p2)*λL
            # Goto 5

        else # Step 4
            y = ifloor(xR-log(v)/λR)
            if y > n
                # Goto 1
                continue
            end
            v *= (u-p3)*λR
            # Goto 5
        end
        
        # Step 5
        # 5.0
        k = abs(y-Mi)
        if (k <= 20) || (k >= nrq/2-1)
            # 5.1
            s = r/q
            a = s*(n+1)
            F = 1.0
            if Mi < y
                for i = (Mi+1):y
                    F *= a/i-s
                end
            elseif Mi > y
                for i = (y+1):Mi
                    F /= a/i-s
                end
            end
            if v > F
                # Goto 1
                continue
            end
            # Goto 6
            return y
        else
            # 5.2
            ρ = (k/nrq)*((k*(k/3.0+0.625)+0.16666666666666666)/nrq+0.5)
            t = -k^2/(2.0*nrq)
            A = log(v)
            if A < t - ρ
                # Goto 6
                return y
            elseif A > t + ρ
                # Goto 1
                continue
            end
            
            # 5.3
            x1 = y+1
            f1 = Mi+1
            z = n+1-Mi
            w = n-y+1

            if A > (xM*log(f1/x1) + (n-M+0.5)*log(z/w) + (y-Mi)log(w*r/(x1*q)) +
                    lstirling(f1) + lstirling(z) + lstirling(x1) + lstirling(w))
                # Goto 1
                continue
            end                
            
            # Goto 6
            return y                        
        end
    end    
end

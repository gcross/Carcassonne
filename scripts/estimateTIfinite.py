from math import cos, pi, sqrt
import sys

N = int(sys.argv[1])
J = float(sys.argv[2])
lam = J/2

if N % 2 == 0:
    m_lo = -N//2
    m_hi =  N//2-1
else:
    m_lo = -(N-1)//2
    m_hi =  (N-1)//2

print(-1/2*sum(sqrt(1+lam**2+2*lam*cos(2*pi*m/N)) for m in range(m_lo,m_hi+1)))

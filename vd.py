from scipy.fft import fft,fftfreq,ifft
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,cos,pi,abs
from matplotlib.pyplot import plot, subplot, grid,show
from matplotlib.pyplot import ylabel, xlabel,stem
# sampling rate
fs = 1000
# sampling interval
ts = 1.0/fs
duration=5
t = np.arange(0,duration,ts)
N= fs*duration
#ngõ vào
def ngovaocos():
    A = float(input("Biên độ: "))
    f = float(input("Tần số: ")) #KHz
    x= A*cos(2*pi*f*t)
    return(x,f)
def ngovaosin():
    A = float(input("Biên độ: "))
    f = float(input("Tần số: ")) #KHz
    x= A*sin(2*pi*f*t)
    return(x,f)
def biendoiFourier(x,n):
    X= np.abs(fft(x))/len(x)
    f= fftfreq(n,1/fs) #Tính tần số ở tâm của mỗi khối fft
    return(X,f)
def BPF(X,fcut):
    #X là tín hiệu cần lọc 
    #f= fftfreq(n,1/fs)
    #fcut = f của tín hiệu + khoảng bảo vệ
    #lọc lần lượt tín hiệu có f tăng dần
    #tín hiệu lọc tiếp theo
    f = fftfreq(len(X), d=1./fs)
    Xl = X.copy()
    Xl[abs(f)>fcut ]=0
    Xlsau= X-Xl
    return(Xl,Xlsau)
def dieucheAM(x,fc):
    A = float(input("Biên độ sóng mang: "))
    songmang = A*cos(2*pi*fc*t)
    xc=[None]*len(songmang)
    for i in range(0,len(xc)):
        xc[i]= songmang[i]*(1+u*x[i])
    return (xc)
u = float(input("nguy ="))
x1,f1= ngovaocos()
fsc1=2*f1
x2,f2=ngovaosin()
xc1= dieucheAM(x1,fsc1)
fsc2 =2*f2 +fsc1+5
xc2= dieucheAM(x2,fsc2)
xc=xc1+xc2
n=2*N
Xc,Fc= biendoiFourier(xc,n)
Xl1,XL1sau= BPF(Xc,fsc1)
plot(Fc,Xl1)
plt.xlim(-200, 200)
show()
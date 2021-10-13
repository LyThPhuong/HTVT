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
tc=np.arange(0,2*duration,ts)
N= fs*duration
# mật độ phổ công suất nhiễu AWGN
Gn= 1e-10
#ngõ vào
def noise():
    A=np.random.normal(0,5)
    f=np.random.normal(0,100)
    dang=np.random.choice(['sin','cos','white'])
    if dang == 'sin':
        nhieu=A*sin(2*pi*f*tc)
        Nd = (A**2)/2
    elif dang == 'cos':
        nhieu=A*cos(2*pi*f*tc)
        Nd = (A**2)/2
    elif dang == 'white':
        nhieu=[1]*len(tc)
        Nd=Gn*2*max(f1,f2)
    return(nhieu,Nd)
def ngovaocos():
    A = float(input("Biên độ: "))
    f = float(input("Tần số: ")) #KHz
    x= A*cos(2*pi*f*t)
    P = (A**2)/2    #công suất TB
    return(x,f,P)
def ngovaosin():
    A = float(input("Biên độ: "))
    f = float(input("Tần số: ")) #KHz
    x= A*sin(2*pi*f*t)
    P = (A**2)/2    #công suất TB
    return(x,f,P)
def biendoiFourier(x,n):
    X= (fft(x))/len(x)
    f= fftfreq(n,1/fs) #Tính tần số ở tâm của mỗi khối fft
    return(X,f)
def BPF(X,fc,f):
    #X là tín hiệu cần lọc 
    #f= fftfreq(n,1/fs)
    #fcut = f của tín hiệu + khoảng bảo vệ
    #lọc lần lượt tín hiệu có f tăng dần
    #tín hiệu lọc tiếp theo
    fx = fftfreq(len(X), d=1./fs)
    Xl = X.copy()
    Xl[abs(fx)< (fc-f)]=0 
    Xl[abs(fx)>(fc+f)]=0
    Xlsau= X-Xl
    return(Xl,Xlsau)
def LPF(X,fcut):
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
def dieucheAM(x,fc,P):
    A = float(input("Biên độ sóng mang: "))
    songmang = A*cos(2*pi*fc*t)
    xc=[None]*len(songmang)
    for i in range(0,len(xc)):
        xc[i]= songmang[i]*(1+u*x[i])
    Sd = ((A**2)*(1+(u**2)*P))/2
    return (xc,A,Sd)
def biendoiFouriernguoc(Xf):
    xc= ifft(Xf)
    return(xc)
def giaidieuche(Xf,fc,Ac):
    songmang = Ac*cos(2*pi*fc*tc)
    yd=  xc *songmang 
    return(yd)


#/////////////////////////////
u = float(input("nguy ="))
print("Hãy nhập các tín hiệu có tần số theo thứ tự tăng dần")
n=2*N
print("Tín hiệu x1(t)")
x1,f1,P1=ngovaocos()
fc1=2*f1
xc1,Ac1,Sd1=dieucheAM(x1,fc1,P1)
Xc1,Fc1=biendoiFourier(xc1,N)
print("Tín hiệu x2(t)")
x2,f2,P2=ngovaosin()
fc2=fc1+ 2*f2 +5
xc2,Ac2,Sd2=dieucheAM(x2,fc2,P2)
Xc2,Fc2=biendoiFourier(xc2,N)
#bo cong
nhieu,Nd=noise()
Sd=Sd1+Sd2
SNR = Sd/Nd
print(SNR)
xc=xc1+xc2+nhieu
Xc,Fc=biendoiFourier(xc,n)
subplot(421)
plot(tc,xc)
subplot(422)
plot(Fc,abs(Xc))
Xl1,Xlsau1=BPF(Xc,fc1,f1)
subplot(423)
plot(Fc,abs(Xl1))
y1=biendoiFouriernguoc(Xl1)
yd1=giaidieuche(y1,fc1,Ac1)
Yd1,Fd1=biendoiFourier(yd1,n)
Yl1,Ylsau1=LPF(Yd1,f1)
Yl1,Ylsau1=BPF(Yl1,f1,0)
subplot(424)
plot(Fd1,abs(Yl1))
Xl2,Xlsau2=BPF(Xlsau1,fc2,f2)
subplot(425)
plot(Fc,abs(Xl2))
y2=biendoiFouriernguoc(Xl2)
yd2=giaidieuche(y2,fc2,Ac2)
Yd2,Fd2=biendoiFourier(yd2,n)
Yl2,Ylsau2=LPF(Yd2,f2)
Yl2,Ylsau2=BPF(Yl2,f2,0)
subplot(426)
plot(Fd2,abs(Yl2))
xs1=biendoiFouriernguoc(Yl1)
xs2=biendoiFouriernguoc(Yl2)
subplot(427)
plot(tc,xs1)
subplot(428)
plot(tc,xs2)
show()

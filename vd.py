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
def LPF(X,fcut):
    #X là tín hiệu cần lọc 
    #f= fftfreq(n,1/fs)
    #fcut = f của tín hiệu + khoảng bảo vệ
    #lọc lần lượt tín hiệu có f tăng dần
    #tín hiệu lọc tiếp theo
    f = fftfreq(len(X), d=1./fs)
    Xl = X.copy()
    Xl[abs(f)<fcut ]=0
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
n=2*N
x1,f1= ngovaocos()
fsc1=2*f1
x2,f2=ngovaosin()
xc1= dieucheAM(x1,fsc1)
fsc2 =2*f2 +fsc1+5
xc2= dieucheAM(x2,fsc2)
Xc1,Fc1=biendoiFourier(xc1,N)
Xc2,Fc2=biendoiFourier(xc2,N)
xc=xc1+xc2
Xc,Fc= biendoiFourier(xc,n)
Xl1,Xl1sau= BPF(Xc,fsc1+f1+4)
Xl2,Xl2sau= BPF(Xl1sau,fsc2+f2+4)
#điều chế xong rồi mới lọc LPF nha 
Xlc1,Xlc2 = LPF(Xc1,f1)
subplot(611)
plot(Fc,Xl1)
ylabel("Xl1(f)")
plt.xlim(-200, 200)
subplot(612)
plot(Fc,Xl2)
ylabel("Xl2(f)")
plt.xlim(-200, 200)
subplot(613)
plot(Fc,Xc)
ylabel("Xc(f)")
plt.xlim(-200, 200)
subplot(614)
plot(Fc1,Xc1)
ylabel("Xc1(f)")
plt.xlim(-200, 200)
subplot(615)
plot(Fc2,Xc2)
ylabel("Xc2(f)")
show()
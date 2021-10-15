from scipy.fft import fft,fftfreq,ifft
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,cos,pi,abs
from matplotlib.pyplot import plot, subplot, grid,show
from matplotlib.pyplot import ylabel, xlabel,stem
# sampling rate
fs = 5000
# sampling interval
ts = 1.0/fs
duration=5
t = np.arange(0,duration,ts)
N= fs*duration
# mật độ phổ công suất nhiễu AWGN
Gn= 1e-10
#ngõ vào
def noise(f,tc):
    A=np.random.normal(0,5)
    fn=np.random.normal(0,100)
    dang=np.random.choice(['sin','cos','white'])
    if dang == 'sin':
        nhieu=A*sin(2*pi*fn*tc)
        Nd = (A**2)/2
    elif dang == 'cos':
        nhieu=A*cos(2*pi*fn*tc)
        Nd = (A**2)/2
    elif dang == 'white':
        nhieu=A*cos(2*pi*0*tc)
        Nd=Gn*2*max(f)
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

def LPF(X,fcut):
    #X là tín hiệu cần lọc 
    #f= fftfreq(n,1/fs)
    #fcut = f của tín hiệu + khoảng bảo vệ
    #lọc lần lượt tín hiệu có f tăng dần
    #tín hiệu lọc tiếp theo
    f = fftfreq(len(X), d=1./fs)
    Xl = X.copy()
    Xl[abs(f)>fcut ]=0
    return(Xl)

def biendoiFouriernguoc(Xf):
    xc= ifft(Xf)
    return(xc)
def giaidieuche(xc,fc):
    songmang =cos(2*pi*fc*tc)
    yd=  xc *songmang 
    return(yd)
def SNR(Sd,Nd):
    return(round(Sd/Nd,3))
def nhapngovao():
    n = int(input("số tín hiệu cần ghép là: "))
    lstngovao=[]
    lsttanso=[]
    lstcongsuat=[]
    for i in range (0,n):
        print("Ngõ vào thứ ",i+1," dạng: ",end='')
        dang = input()
        if dang=="cos":
            x,f,P = ngovaocos()
        elif dang=="sin":
            x,f,P =ngovaosin()
        lstngovao.append(x)
        lsttanso.append(f)
        lstcongsuat.append(P)
    return(lstngovao,lsttanso,lstcongsuat,n)
def songmang(f):
    fc=[]
    fc.append(2*f[0])
    for i in range(1,len(f)):
        fc.append(fc[i-1]+2*f[i]+5)
    return(fc)
def dieucheAM(x,fc,P):
    A = float(input("Biên độ sóng mang: "))
    songmang = A*cos(2*pi*fc*t)
    xc=[None]*len(songmang)
    for i in range(0,len(xc)):
        xc[i]= songmang[i]*(1+u*x[i])
    Sd = ((A**2)*(1+(u**2)*P))/2
    return (xc,Sd)
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
    return(Xl)
#main
u = float(input("Hệ số điều chế u="))
x,f,P,n=nhapngovao()
fc=songmang(f)
mc=[None]*len(x)
Pd=[None]*len(P)
for i in range(0,len(x)):
    mc[i],Pd[i]=dieucheAM(x[i],fc[i],P[i])
Sd=sum(Pd)
xc=[]
tc=np.arange(0,n*duration,ts)
nhieu,Nd=noise(f,tc)
for i in range(0,len(mc)):
    xc+=mc[i]
xc+=nhieu
Nc=n*N
Xc,Fc=biendoiFourier(xc,Nc)
#Xc=list(Xc)
print(type(Xc),len(Xc))
s=[]
for i in range(0,n):
    Xl=BPF(Xc,fc[i],f[i])
    y=biendoiFouriernguoc(Xl)
    yd=giaidieuche(y,fc[i])
    Yd,Fd=biendoiFourier(yd,Nc)
    Yl=LPF(Yd,f[i])
    s.append(biendoiFouriernguoc(Yl))
for i in range(0,len(s)):
    subplot(len(s),1,i+1)
    plot(tc,s[i])
SNR=SNR(Sd,Nd)
print("SNR=",SNR)
show()
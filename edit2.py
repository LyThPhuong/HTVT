import numpy as np
import matplotlib as plt
from matplotlib.pyplot import plot, subplot, grid,show
from matplotlib.pyplot import ylabel, xlabel
from numpy import sin,cos,pi,arctan
from scipy.fft import fft,fftfreq

f_laymau= 100 #KHz #số điểm tín hiệu sử dụng biểu diễn sóng
duration = 10  #ms #Chiều dài của mẫu được tạo
N= f_laymau*duration
t=np.linspace(-duration,duration,f_laymau*duration)
#ham sinc
def ngovaosinc():
    A = float(input("Biên độ: "))
    f = float(input("Tần số: ")) #KHz
    wt= 2*pi*f*t
    x= A*np.sinc(wt)
    return(x,f)
def ngovaoxungvuong():
    A = float(input("Biên độ: "))
    f = float(input("Tần số: ")) 
    T=2/f
    Pi=[]
    for i in t:
        if i >= -T/2 and i <= T/2:
            Pi.append(A)
        else:
            Pi.append(0)
    return(Pi,f)
def ngovaotamgiac():
    A = float(input("Biên độ: "))
    f = float(input("Tần số: ")) 
    tg=[]
    T=2/f
    a =-2*A/T
    b= A
    for i in t:
        if i >=0 and i <=T/2:
            tg.append(a*i+b)
        elif i>= -T/2 and i<=0:
            tg.append(-a*i +b)
        else:
            tg.append(0)
    return(tg,f)
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
def biendoiFourier(xc,n):
    Xf= np.abs(fft(xc))/len(xc)
    f= fftfreq(n,1/f_laymau) #Tính tần số ở tâm của mỗi khối fft
    return(Xf,f)
def dieucheAM(x,fc):
    A = float(input("Biên độ sóng mang: "))
    #fsc = int(input("Tần số sóng mang: ")) #KHz
    t= np.linspace(0,duration,f_laymau*duration)
    songmang = A*cos(2*pi*fc*t)
    xc=[None]*len(songmang)
    for i in range(0,len(xc)):
        xc[i]= songmang[i]*(1+u*x[i])
    return (xc)

#dieuchexungvuong
u = float(input("nguy ="))
x1,f1= ngovaocos()
fsc= 2*f1
xc1=dieucheAM(x1,fsc)
#dieuchesongsin
x2,f2=ngovaosin()
fsc += 2*f2+5
xc2 =dieucheAM(x2,fsc)
'''#dieuchesongcos
x3,f3=ngovaocos()
fsc += 2*f3+5
xc3 =dieucheAM(x3,fsc)
#dieuchexungtamgiac
x4,f4=ngovaosin()
fsc += 2*f4+5
xc4 =dieucheAM(x4,fsc)'''
xc=xc1+xc2#+xc3+xc4
Xc1,fc11= biendoiFourier(xc1,N)
Xc2,fc12= biendoiFourier(xc2,N)
n= 2*N
Xc,fc1=biendoiFourier(xc,n)
Xcc= Xc1+Xc2
subplot(4,1,3)
plot(fc11,Xc1)
ylabel("Xc1(f)")
subplot(4,1,2)
plot(fc12,Xc2)
ylabel("Xc2(f)")
subplot(4,1,1)
plot(fc1,Xc)
ylabel("Xc(f)=biendoifourier của xc1+xc2")
subplot(4,1,4)
plot(fc11,Xcc)
ylabel("Xc(f)=Xc1+Xc2")
'''sinx,f2= ngovaosin()
x=sinx
fsc=fsc1 + f1 + 5
xc2=dieucheAM(x)
#dieuchesongcos
cosx,f3= ngovaocos()
x=cosx
fsc=fsc + f2 + 5
xc3=dieucheAM(x)
#dieuchexungtamgiac
x1,f4= ngovaotamgiac()
x=x1
fsc=fsc + f3 + 5
xc4=dieucheAM(x)
#tinhieuFDM
xc = xc1 + xc2 + xc3
print(len(xc))
xf1,f1 = biendoiFourier(xc)
plot(xf1)'''
show()
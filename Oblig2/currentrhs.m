function r=currentrhs(t,h)
global dx Bo grav

hx=currdiff1(h,dx);

hxxx=diff2(hx,dx);

s=currdiff1(-(1/(Bo))*h.^3.*hxxx+(grav)*hx.*h.^3,dx);

r=s;
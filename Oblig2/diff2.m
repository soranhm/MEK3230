function dy=diff2(y,dx)
l=length(y);
i=1:l;
im1=mod(i-2,l)+1;
ip1=mod(i,l)+1;
im2=mod(i-3,l)+1;
ip2=mod(i+1,l)+1;
s=zeros(1,l);
s=(y(ip1)-2*y(i)+y(im1))./(dx.^2);
dy=s;

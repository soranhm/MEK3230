% OPPGAVE 2.1 - 1)

n = 20;
r = linspace(1,1000);
m = 1; gamma = 1; 
w0 = linspace(0.5,2,n);

for i = 1:n
    kons = (2*pi*w0(i))/m;
    polar(kons + (gamma/m)*log(r),r);
    hold on
end

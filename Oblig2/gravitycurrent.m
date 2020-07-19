clear all
close all
%main file
global dx Bo grav
%%%%%%mesh%%%%%%%%
L=6*pi;%st?rrelsen p? domenet
n=600+1;%gridpunkt
dx=L/(n-1);%gridst?rrelse
x=-L/2:dx:L/2;%x-koordinat
%%%%%%%%%%%%%%%

%%%%parameters%%%%
%%%%%%%%%%%%%%%%%%%
Bo=1.0E19;%Bond tall
grav=1.0;%retningen p? gravitasjonskraften
%%%%%%%%%%%%%%%%%%
h0=(1-(tanh(2*(x-0*3*pi)).^2))+0.0;

figure(1919)
set(gca,'Fontname','Times New Roman','FontSize',30)
plot(x,h0,'LineWidth',3)
xlabel('X')
ylabel('H_0(X,T=0)')
title('Initial condition')

q=sum(h0)*dx;
options = odeset('RelTol',1e-6,'AbsTol',1e-6,'InitialStep', 1.0e-4);
[t,H] = ode45(@currentrhs,[0:500:8E4],h0,options);
%tids integrator + numerisk diskretisering av ligningen 
%t=tid, H=H(X,t)=overflaten   

    for i=2:length(H(:,1))
        set(gca,'Fontname','Times New Roman','FontSize',30)
        ti = t(i);
        T = ti/((3*L^2*1)/(grav*1*max(h0)^3));
        plot(x/(T.^(1/5)),H(i,:)/(T^((-1/5))),'LineWidth',3)
        hold on
        xlabel('X/Xn(T)')
        ylabel('H(X,T)/Ht(T)')
        title('Plot av skalering')
        axis([-3*pi 3*pi 0 0.2])
        pause(0.1)
    end
  
h = zeros(1,161);
    for i = 1:161;
        h(i) = max(H(i,:));
    end
figure()
loglog(t.^(-1/5))
hold on
loglog(h)
title('Plott av Ht(T) og T^(-1/5)')
legend('Ht(T)','T^[-1/5]')
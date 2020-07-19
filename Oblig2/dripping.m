clear all
close all
%main file
global dx Bo grav
%%%%%%mesh%%%%%%%%
L=2*pi;%st?rrelsen p? domenet
n=400+1;%gridpunkt
dx=L/(n-1);%gridst?rrelse
x=-L/2:dx:L/2;%x-koordinat
grav=-1.0;%retningen p? gravitasjonskraften
%%%%%%%%%%%%%%%

%%%%parameters%%%%
%%%%%%%%%%%%%%%%%%%
Bo=1;%Bond tall
ki=(1.1)^2;%b?lgetallet til forstyrrelsen/perturbasjonen
ki2=(0.9)^2;%b?lgetallet til forstyrrelsen/perturbasjonen
%%%%%%%%%%%%%%%%%%
h0=1+0.005*sin(ki*x);
h02=1+0.005*sin(ki2*x);


figure(1919)
set(gca,'Fontname','Times New Roman','FontSize',30)
plot(x,h0,'LineWidth',3)
xlabel('X')
ylabel('H_0(X,T=0)')
title('Initial condition')

options = odeset('RelTol',1e-8,'AbsTol',1e-8,'InitialStep', 1.0e-4);
[t,H] = ode15s(@currentrhs,[0 600],h0,options);
[t2,H2] = ode15s(@currentrhs,[0 600],h02,options);
%tids integrator + numerisk diskretisering av ligningen 
%t=tid, H=H(X,t)=overflaten   

    for i=2:length(H2(:,1))
        T = 600;
        plot(x/(T^(1/5)),H(i,:)/(T^(-1/5)),'g',x/(T^(1/5)),H2(i,:)/(T^(-1/5)),'r','LineWidth',3)
        xlabel('X')
        ylabel('H(X,T)')
        legend('Stabil','Ustabil')
      %  axis([-3*pi 3*pi 0 0.2])
        pause(0.01)
    end
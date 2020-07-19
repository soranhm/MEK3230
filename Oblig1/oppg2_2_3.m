% OPPGAVE 2.2 - 3)

g = 9.81; gamma = 1; n = 100;
rho = 1; p0 = 1; a = 1;
r = linspace(0,2,n);
z = zeros(1,n);

for i = 1:n
    if r(i) > a
        z(i) = (gamma^2 * a^2)/g - (gamma^2 * a^4)/(2*g*r(i)^2) - p0/(rho*g);
    elseif r == a
        z(i) = - p0/(g*rho) + (gamma^2*a^2)/g
    else
        z(i) = (gamma^2 * r(i)^2)/(2*g) - p0/(rho*g);
    end
end

plot(r,z)


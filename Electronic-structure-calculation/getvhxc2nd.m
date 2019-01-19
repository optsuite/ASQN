function hxc = getvhxc2nd(rho)

itny = find(rho<1e-16);
if (~isempty(itny))
    rho(itny) = 1e-16;
end;

rs    = (4*pi*rho/3).^(-1/3);
ibig  = find(rs >= 1);
isml  = find(rs <  1);
%
cex  = -1.969490099;
A1 = -0.096;
A2 = -0.0232;
A3 = 0.0622;
A4 = 0.004;
%
B1 = -0.2846;
B2 = 1.0529;
B3 = 0.3334;

rs1 = rs(isml);
rs2 = rs(ibig);
lnrs1 = log(rs1);

hhvx = (cex/3)*rho.^(-2/3);

drs = (-4*pi/9)*(4*pi*rho/3).^(-4/3);
hvc = zeros(size(rho));
hvc(isml) = (2/3)*A2 + A4/3 + A3./rs1 + (2/3)*A4*lnrs1;
hvc(ibig) = -( (B1*(7*B2^2 + 8*B3))/12 + ...
    (B1*(5*B2 + 16*B3^2*rs2.^(3/2) + 21*B2*B3*rs2))./(12*rs2.^(1/2))...
    )./(B3*rs2 + B2*rs2.^(1/2) + 1).^3;
hvc = hvc.*drs;
hxc = (hhvx + hvc)/2;
%hxc = hvc/2;
%hxc = hhvx/2;

end
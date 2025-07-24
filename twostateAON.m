%%Function for fitting melting curves to two-state all-or-nothing model
%%Order of Parameters for P0 [H, Tm, Cp, dm, db, mm, mb]
function [ Out ] = twostateAON(T,Data,p0,R,C,selfcomp,bimol,heatcap)

if heatcap == false
    H = p0(1);
    Tm = p0(2);
    dm = p0(3);
    db = p0(4);
    mm = p0(5);
    mb = p0(6);
else
    H = p0(1);
    Tm = p0(2);
    Cp = p0(3);
    dm = p0(4);
    db = p0(5);
    mm = p0(6);
    mb = p0(7);    
end    

if bimol == true
if heatcap == true
if selfcomp == true
    S = (H+R.*Tm.*log(C./4))./Tm;
    G = H-T.*S+Cp.*(T-Tm-T.*log(T./Tm));
    K = exp(-G./(R.*T));
    f = (C+K-sqrt(K.^2+2.*C.*K))./(C); % Fraction of intact base pairs
    Sd = dm.*T+db; %dimer
    Sm = mm.*mb; %monomer
    Vfit = f.*(Sd-Sm)+Sm;
else
    S = (H+R.*Tm.*log(C./4))./Tm;
    G = H-T.*S+Cp.*(T-Tm-T.*log(T./Tm));
    K = exp(-G./(R.*T));
    f = (C+K-sqrt(K.^2+2.*C.*K))./(C); % Fraction of intact base pairs
    Sd = dm.*T+db; %dimer
    Sm = mm.*T+mb; %monomer
    Vfit = f.*(Sd-Sm)+Sm;  
end
else
if selfcomp == true
    S = (H+R.*Tm.*log(C))./Tm;
    G = H-T.*S;
    K = exp(-G./(R.*T));
    f = (4.*C+K-sqrt(K.^2+8.*C.*K))./(4.*C); % Fraction of intact base pairs
    Sd = dm.*T+db; %dimer
    Sm = mm.*T+mb; %monomer
    Vfit = f.*(Sd-Sm)+Sm;
else
    S = (H+R.*Tm.*log(C./4))./Tm;
    G = H-T.*S;
    K = exp(-G./(R.*T));
    f = (C+K-sqrt(K.^2+2.*C.*K))./(C); % Fraction of intact base pairs
    Sd = dm.*T+db; %dimer
    Sm = mm.*T+mb; %monomer
    Vfit = f.*(Sd-Sm)+Sm; 
end
end
else
if heatcap == true
    S = H./Tm;
    G = H-T.*S+Cp.*(T-Tm-T.*log(T./Tm));
    K = exp(-G./(R.*T));
    f = 1./(K+1);
    Sd = dm.*T+db;
    Sm = mm.*T+mb;
    Vfit = f.*(Sd-Sm)+Sm;  
else
    S = H./Tm;
    G = H-T.*S;
    K = exp(-G./(R.*T));
    f = 1./(K+1);
    Sd = dm.*T+db;
    Sm = mm.*T+mb;
    Vfit = f.*(Sd-Sm)+Sm;
end
end

Out = Data-Vfit;
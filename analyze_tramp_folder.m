function res = analyze_tramp_folder(path, temps)
%% given wn, abs output the normalized wn, abs
% Usage:
%   res = analyze_tramp_folder("C:\...\folder", 20:60);

if nargin < 2
    temps = 20:1:60;    
end


% --------- Normalize every file and store each 2d array in a cell array
oldpwd = pwd;
cleanupObj = onCleanup(@() cd(oldpwd));
cd(path);

files = dir(fullfile(path, '*'));
multiArray = {}; %cell(length(files), 1);
for k = 1:length(files)
    if files(k).isdir
        continue;
    end
    fileName = fullfile(path, files(k).name);
    data = load(fileName);

    wn  = data(:,1);
    abs = data(:,2);

    [wn2, abs2] = norm_local(wn, abs);

    arr = [wn2, abs2];

    multiArray{end+1} = arr;
end

% --------- Plots all spectra in multiArray together
sample       = multiArray(1,1);
sample_data  = sample{1,1};
x            = sample_data(:,1);
y            = sample_data(:,2);
figure;
plot(x,y);
title("Normalized & baseline corrected Spectra"); % (title text kept)

hold on;
for i = 2:length(multiArray)
    lsample      = multiArray(1,i);
    lsample_data = lsample{1,1};
    x = lsample_data(:,1);
    y = lsample_data(:,2);
    plot(x,y);
end
hold off;

% --------- SVD
% stack columns
s = [];
for i = 1:length(multiArray)
    s(:,i) = multiArray{i}(:,2);
end

% find row index for 1600, 1700 wn
WN_i = find(wn >= 1585, 1, 'first');
WN_f = find(wn <= 1700, 1, 'last');

subS = s(WN_i:WN_f, :);

figure;
for i = 1:size(s,2)
    plot(wn(WN_i:WN_f), subS(:, i));
    hold on;
end
title('AM1 Peak');
hold off;

[U, S, V] = svd(subS);
% only do this for the am1 peak 1600-1700 wn

dS    = diag(S);
index = 1:1:length(dS);
figure;
plot(index, dS);
title("basis weights");

figure;
subplot(2, 2, 1);
plot (wn(WN_i:WN_f), U(:,1));
title('first spectral component-unweighted');

subplot(2,2,2);
plot(temps, V(:,1));
title('first temperature component-unweighted');

% plot ~ first 4 spectral components & first 4 temp components
subplot(2,2,3);
for i = 1:4
    plot(wn(WN_i:WN_f), dS(i,1)*U(:,i) );
    hold on;
end
hold off;
title('weighted spectral cols 1-4');

subplot(2,2,4);
for i = 1:4
    plot(temps, dS(i,1)*V(:,i) );
    hold on;
end
hold off;
title('weighted temp cols 1-4');

% --------- Two-state fit section (kept as-is)
addpath('C:\Users\ibenderskii\Documents\MATLAB\');  % your line
% guess parameters
H  = 4e5;
R  = 8.3144;
Tm = 37+273.15 ;
T  = temps + 273.15;
dm = -0.03;
db = 10 ;
mm = -0.03;
mb = 9.5;

Sx = H./Tm;
G  = H - T.*Sx;
K  = exp(-G./(R.*T));
f  = 1./(K+1);
Sd = dm.*T + db;
Sm = mm.*T + mb;
Vtest = f.*(Sd-Sm) + Sm;

figure;
plot(T, Vtest);

p0(1)= H;
p0(2)= Tm;
p0(3)= dm ;
p0(4)= db;
p0(5)= mm;
p0(6)= mb;

lb = ones(length(p0),1).*-10;
ub = ones(length(p0),1) .*Inf;

Data     = V(:,2);
normData = Data - min(Data);
normData = normData.*(1/max(normData));

func = @(p0) twostateAON(T, normData.', p0, R, 0, false, false, false);
p    = lsqnonlin(func, p0, lb, ub);

H  = p(1);
Tm = p(2);
dm = p(3);
db = p(4);
mm = p(5);
mb = p(6);

Sx = H./Tm;
G  = H - T.*Sx;
K  = exp(-G./(R.*T));
f  = 1./(K+1);
Sd = dm.*T + db;
Sm = mm.*T + mb;
Vfit = f.*(Sd-Sm) + Sm;

figure;
plot(T, normData);
hold on;
plot(T, Vfit);
hold off;




%% ===== Transition temperature extraction =====

% 1) Choose an observable Y(T)
%    (a) PC1 score (weighted): Y = sigma1 * V(:,1)
Y_pc1 = S(1,1) * V(:,1);

%    (b) Or band area in the same window for each spectrum (columns of subS)
%       This is optional; uncomment if you prefer band area route:
% Y_area = sum(subS, 1).';   % column sum -> (nTemps x 1)

% Select which observable to use
Y = Y_pc1;           % or Y = Y_area;
Y = Y(:);
temps = temps(:);
if numel(Y) ~= numel(temps)
    error('Mismatch: %d spectra vs %d temps', numel(Y), numel(temps));
end
% 2) Normalize to [0,1] for stable fitting/interpretation
Y = (Y - min(Y)) / max(1e-12, (max(Y) - min(Y)));

% 3) Optional smoothing to reduce noise (keeps shape; tune window)
Y_s = smoothdata(Y, 'sgolay', 7);

% 4) Sigmoid (two-state/Boltzmann) fit to get midpoint and width
%    model: y = A2 + (A1 - A2)/(1 + exp((T - Tm)/dT))
boltz = @(b, T) b(2) + (b(1) - b(2)) ./ (1 + exp((T - b(3)) / b(4)));
% Initial guesses:
[~, iMaxSlope] = max(abs(gradient(Y_s, temps)));
Tm0   = temps(iMaxSlope);
dT0   = (max(temps)-min(temps))/20;  % initial width guess
A10   = 1; A20 = 0;
b0    = [A10, A20, Tm0, dT0];

% Bounds (A1,A2 within [0,1]; width positive but not huge)
lb = [0, 0, min(temps)-5, 0.1];
ub = [1, 1, max(temps)+5, (max(temps)-min(temps))];

opts = optimoptions('lsqcurvefit', 'Display','off');
try
    bfit = lsqcurvefit(boltz, b0, temps(:), Y_s(:), lb, ub, opts);
catch
    % Fallback if Optimization Toolbox unavailable: use nlinfit if you have Statistics Toolbox
    % bfit = nlinfit(temps(:), Y_s(:), @(b,T) boltz(b,T), b0);
    % Otherwise, just use derivative/half-height methods below.
    bfit = b0;
end
Y_fit = boltz(bfit, temps(:));
Tm_fit = bfit(3);      % sigmoid midpoint
dT_fit = abs(bfit(4)); % transition width (cooperativity proxy)

% 5) Derivative-based estimate: max slope temperature
dYdT  = gradient(Y_s, temps);
[~, iDerivMax] = max(abs(dYdT));
Tm_deriv = temps(iDerivMax);

% 6) Half-height estimate: where Y crosses 0.5
Tm_half = interp1(Y_s, temps, 0.5, 'linear', 'extrap');

% 7) Plot quick diagnostic (optional)
figure; 
subplot(1,2,1);
plot(temps, Y, 'o', 'DisplayName','Y raw'); hold on;
plot(temps, Y_s, '-', 'DisplayName','Y smooth');
plot(temps, Y_fit, '--', 'DisplayName','Boltzmann fit');
xlabel('T (°C)'); ylabel('Observable (norm.)'); grid on;
legend('Location','best');
title(sprintf('Tm_{fit}=%.2f, Tm_{dY/dT}=%.2f, Tm_{1/2}=%.2f', Tm_fit, Tm_deriv, Tm_half));

subplot(1,2,2);
plot(temps, dYdT, '-o'); grid on;
xlabel('T (°C)'); ylabel('dY/dT');
title('Derivative (steepest point ~ T_m)');

% 8) Save results to a struct you return (or to disk)

% --------- Package outputs (optional)
res = struct();
res.path       = path;
res.files      = {files(~[files.isdir]).name}';
res.wn         = wn;
res.multiArray = multiArray;
res.U = U; res.S = S; res.V = V; res.dS = dS;
res.WN_i = WN_i; res.WN_f = WN_f;
res.temps = temps;
res.fit.params = p;
res.fit.Vfit   = Vfit;
res.fit.normData = normData;
res.Tm_fit   = Tm_fit;
res.dT_fit   = dT_fit;
res.Tm_deriv = Tm_deriv;
res.Tm_half  = Tm_half;
res.method   = 'PC1';
end

% ==================== Local function (your original 'norm') ====================
function [xf,yf] = norm_local(x, y)
% Your code unchanged except the function name.
wn  = x;
absorb = y;

wn_i = find(wn >= 4000, 1, 'first');
wn_f = find(wn <= 7000, 1, 'last');

absorb_range = absorb(wn_i : wn_f);
wn_range  = wn(wn_i : wn_f);

% polynomial fit
p    = polyfit(wn_range, absorb_range, 1);
line = p(1).*wn + p(2);
absorb2 = absorb - line;

% normalize
wn2_i = find(wn >= 1600, 1, 'first');
wn2_f = find(wn <= 1700, 1, 'last');

absorb2_range = absorb2(wn2_i : wn2_f) ;
if sum (absorb2_range) < 0
        %absorb2 = -absorb2;
        absorb2_range = -absorb2_range;
end

N   = sum(absorb2_range);
absorb = absorb2 ./ N;

xf = wn;
yf = absorb;
end

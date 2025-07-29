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
WN_f = find(wn >= 1700, 1, 'first');

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
end

% ==================== Local function (your original 'norm') ====================
function [xf,yf] = norm_local(x, y)
% Your code unchanged except the function name.
wn  = x;
abs = y;

wn_i = find(wn >= 4000, 1, 'first');
wn_f = find(wn >= 7000, 1, 'first');

abs_range = abs(wn_i : wn_f);
wn_range  = wn(wn_i : wn_f);

% polynomial fit
p    = polyfit(wn_range, abs_range, 1);
line = p(1).*wn + p(2);
abs2 = abs - line;
abs  = abs2;

% normalize
wn2_i = find(wn >= 1600, 1, 'first');
wn2_f = find(wn >= 1700, 1, 'first');

abs2_range = abs2(wn2_i : wn2_f);
wn2_range  = wn(wn2_i : wn2_f);

N   = sum(abs2_range);
abs = abs2 ./ N;

xf = wn;
yf = abs;
end

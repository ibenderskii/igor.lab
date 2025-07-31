function res = simple_tm_from_folder(path, temps, varargin)
% SIMPLE_TM_FROM_FOLDER  Get transition temperature from a T-ramp folder
% Usage:
%   res = simple_tm_from_folder("C:\...\folder", 20:60);
%
% Inputs
%   path  : folder with text files, each file has [wn, absorbance] columns
%   temps : vector of temperatures corresponding to files (same length as files)
%
% Optional name-value:
%   'Win'       : [lo hi] wavenumber window to integrate (default [1585 1700])
%   'Baseline'  : [lo hi] wavenumber range for linear baseline (default [4000 7000])
%   'SmoothN'   : odd integer window for smoothing Y (default 5)
%
% Output (struct)
%   res.temps     : sorted temperatures used (°C)
%   res.Y         : normalized observable in [0,1]
%   res.Ys        : smoothed Y
%   res.dYdT      : numerical derivative dY/dT
%   res.Tm_half   : half-height temperature (Y=0.5)
%   res.Tm_deriv  : temperature at maximum slope
%   res.width     : FWHM of derivative (optional width measure)
%   res.files     : file names used
%   res.win, res.baseline : ranges used

% --- defaults & args ---
p = inputParser;
p.addParameter('Win', [1585 1700], @(v)isnumeric(v)&&numel(v)==2);
p.addParameter('Baseline', [6000 7000], @(v)isnumeric(v)&&numel(v)==2);
p.addParameter('SmoothN', 7, @(v)isnumeric(v)&&isscalar(v)&&v>=1);
p.parse(varargin{:});
win       = sort(p.Results.Win);
baseRange = sort(p.Results.Baseline);
smoothN   = p.Results.SmoothN;

% --- list files ---
d = dir(path);
d = d(~[d.isdir]);            % files only
if isempty(d)
    error('No files found in %s', path);
end
% Sort by name to have a stable order
[~,ord] = sort({d.name});
d = d(ord);

if nargin < 2 || isempty(temps)
    % if temps not provided, try to infer a linear ramp
    temps = linspace(20,60,numel(d));
end
if numel(temps) ~= numel(d)
    error('Number of temps (%d) must match number of files (%d).', numel(temps), numel(d));
end

areas = nan(numel(d),1);
files = strings(numel(d),1);

for i = 1:numel(d)
    files(i) = string(d(i).name);
    M = load(fullfile(path, d(i).name));
    if size(M,2) < 2
        warning('File %s does not have 2 columns. Skipping.', d(i).name);
        continue
    end
    wn  = M(:,1);
    a   = M(:,2);

    % Ensure column vectors
    wn = wn(:); a = a(:);

    % Flip to ascending wavenumber
    if wn(1) > wn(end)
        wn = flipud(wn); a = flipud(a);
    end

    % Linear baseline using a broad high-frequency region (if present)
    idxb = wn>=baseRange(1) & wn<=baseRange(2);
    if sum(idxb) >= 2
        pbl = polyfit(wn(idxb), a(idxb), 1);
        a2  = a - (pbl(1)*wn + pbl(2));
    else
        a2 = a;
    end

    % Integrate band area in chosen window
    idxw = wn>=win(1) & wn<=win(2);
    if ~any(idxw)
        warning('No points in integration window for %s. Skipping.', d(i).name);
        continue
    end
    areas(i) = trapz(wn(idxw), a2(idxw));
end

% Pack valid points
ok = isfinite(areas) & isfinite(temps(:));
areas = areas(ok);
temps = temps(:);
temps = temps(ok);
files = files(ok);

% Sort by temperature
[temps, idx] = sort(temps);
areas = areas(idx);
files = files(idx);

% Normalize to [0,1]
rngA = max(areas) - min(areas);
if rngA < eps
    % Flat signal; cannot define a transition robustly
    res = struct('temps',temps,'Y',nan(size(temps)),'Ys',nan(size(temps)), ...
                 'dYdT',nan(size(temps)),'Tm_half',NaN,'Tm_deriv',NaN, ...
                 'width',NaN,'files',files,'win',win,'baseline',baseRange);
    return
end
Y = (areas - min(areas)) / rngA;

% Make it increase with T (flip if decreasing)
if mean(diff(Y)) < 0
    Y = 1 - Y;
end

% Smooth (simple moving average or savgol)
if smoothN > 1
    smoothN = min(smoothN, numel(Y) - (1-mod(numel(Y),2)));  % odd-ish, <= length
    if smoothN >= 3
        Ys = smoothdata(Y, 'movmean', smoothN);
    else
        Ys = Y;
    end
else
    Ys = Y;
end

% Derivative and Tm by derivative
dYdT = gradient(Ys, temps);
[~, iMax] = max(dYdT);             % since Y increases, use max (not abs)
Tm_deriv  = temps(iMax);

% Half-height (Y=0.5), use monotone envelope to be safe
Ymono = cummax(Ys);                 % non-decreasing
[Yu, iu] = unique(Ymono);
Tu = temps(iu);
if numel(Yu) >= 2 && Yu(1) <= 0.5 && Yu(end) >= 0.5
    Tm_half = interp1(Yu, Tu, 0.5, 'pchip','extrap');
else
    Tm_half = NaN;
end

% Width estimate: derivative FWHM
w = NaN;
if numel(dYdT) >= 3 && any(dYdT>0)
    half = 0.5*max(dYdT);
    % left crossing
    iL = find(dYdT(1:iMax) >= half, 1, 'first');
    iR = find(dYdT(iMax:end) >= half, 1, 'last') + iMax - 1;
    if ~isempty(iL) && ~isempty(iR) && iR > iL
        w = temps(iR) - temps(iL);
    end
end

% Return results
res = struct();
res.temps    = temps;
res.Y        = Y;
res.Ys       = Ys;
res.dYdT     = dYdT;
res.Tm_half  = Tm_half;
res.Tm_deriv = Tm_deriv;
res.width    = w;
res.files    = files;
res.win      = win;
res.baseline = baseRange;

% Quick diagnostic plot (optional)
if nargout == 0
    figure('Color','w');
    subplot(1,2,1);
    plot(temps, Y, 'o', temps, Ys, '-', 'LineWidth',1.5); grid on;
    hold on; yline(0.5,'--'); xline(Tm_half,':r','Tm_{1/2}');
    xlabel('T (°C)'); ylabel('Y (norm)'); legend('Y','Ys','0.5','Tm_{1/2}');
    title(sprintf('Half-height T_m = %.2f °C', Tm_half));

    subplot(1,2,2);
    plot(temps, dYdT, '-o'); grid on; hold on;
    xline(Tm_deriv,':r','Tm_{deriv}');
    xlabel('T (°C)'); ylabel('dY/dT');
    title(sprintf('Derivative T_m = %.2f °C (width ≈ %.2f °C)', Tm_deriv, w));
end
end

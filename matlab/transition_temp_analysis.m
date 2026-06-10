

base = "C:\Users\ibend\data\PNIPAM_300kda_20_1_60_T_Ramps";
folders = dir(base);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name},{'.','..'}));

Tms_half  = nan(numel(folders),1);
Tms_deriv = nan(numel(folders),1);
widths    = nan(numel(folders),1);
concs     = nan(numel(folders),1);

for k = 1:numel(folders)
    fpath = fullfile(base, folders(k).name);
    
    d = dir(fpath); d = d(~[d.isdir]);
    temps = linspace(20,60,numel(d));      
    
    try
        res = simple_tm_from_folder(fpath, temps);
        Tms_half(k)  = res.Tm_half;
        Tms_deriv(k) = res.Tm_deriv;
        widths(k)    = res.width;
        concs(k)     = parseConcFromName(folders(k).name);
    catch ME
        warning('Folder "%s" failed: %s', folders(k).name, ME.message);
    end
end

ok = isfinite(concs) & (isfinite(Tms_half) | isfinite(Tms_deriv));
concs  = concs(ok);
Tms_half  = Tms_half(ok);
Tms_deriv = Tms_deriv(ok);
widths    = widths(ok);

[concs, idx] = sort(concs);
Tms_half  = Tms_half(idx);
Tms_deriv = Tms_deriv(idx);
widths    = widths(idx);

figure; 
subplot(1,2,1); plot(concs, Tms_half, 'o-', concs, Tms_deriv, 's-'); 
xlabel('wt%'); ylabel('T_{tr} (°C)'); legend('half-height','derivative'); grid on;
subplot(1,2,2); plot(concs, widths, 'o-'); xlabel('wt%'); ylabel('width (°C)'); grid on;

function c = parseConcFromName(nameStr)
    % Try to extract the first decimal/float-looking number
    % Accepts "0.25wt", "1 wt%", "7.5", "10%", "1_wt", etc.
    % Returns NaN if nothing is found.
    if ~ischar(nameStr) && ~isstring(nameStr), c = NaN; return; end
    nameStr = char(nameStr);
    tok = regexp(nameStr, '([\d]+(?:\.\d+)?)', 'tokens', 'once');
    if isempty(tok)
        c = NaN;
    else
        c = str2double(tok{1});
    end
end
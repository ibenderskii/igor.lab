clear
clc;  % optional


disp('Starting…');
base = "C:\Users\ibend\data\PNIPAM_300kda_20_1_60_T_Ramps";
folders = dir(base);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name},{'.','..'}));

allres = cell(numel(folders),1);

Tms  = nan(numel(folders),1);
dTs  = nan(numel(folders),1);
concs = nan(numel(folders),1);


for k = 1:numel(folders)
    fpath = fullfile(base, folders(k).name);
    fprintf('Processing: %s\n', fpath);

    concs(k) = parseConcFromName(folders(k).name);
    try
        allres{k} = analyze_tramp_folder(fpath, 20:60);
        Tms(k) = allres{k}.Tm_fit;
        dTs(k) = allres{k}.dT_fit;
    catch ME 
        warning('you dumbass this folder"%s" failed: %s', folders(k).name, ME.message);
    end
end

ok = isfinite(concs) & isfinite(Tms) & isfinite(dTs);
concs = concs(ok);  Tms = Tms(ok);  dTs = dTs(ok);

[concs, idx] = sort(concs);
Tms = Tms(idx);
dTs = dTs(idx);

figure; 
subplot(1,2,1); plot(concs, Tms, 'o-'); xlabel('wt%'); ylabel('T_{tr} (°C)'); grid on;
subplot(1,2,2); plot(concs, dTs, 'o-'); xlabel('wt%'); ylabel('width dT (°C)'); grid on;

summaryT = table(concs(:), Tms(:), dTs(:), 'VariableNames', {'wtpct','Ttr_C','width_C'});
writetable(summaryT, fullfile(base, 'summary_transition_vs_conc.csv'));

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
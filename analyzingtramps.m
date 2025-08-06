clear
close all


disp('Starting…');
base =  "C:\Users\ibend\data\PNIPAM_300kda_20_1_60_T_Ramps";
folders = dir(base);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name},{'.','..'}));

vfits = cell(numel(folders),1);
vdatas = cell(numel(folders),1);
temps = 20:60;
Tms =  zeros(numel(folders),1);

concs = nan(numel(folders),1);
addpath("C:\Users\ibend\igor.lab");

for k = 1:numel(folders)
    fpath = fullfile(base, folders(k).name);
    fprintf('Processing: %s\n', fpath);
    
    concs(k) = parseConcFromName(folders(k).name);
    try
        res =analyze_tramp_folder(fpath, temps, concs(k));
        vfits{k} = res.ndata;
        vdatas{k} = res.Vmatfit; 
        Tms(k,1) = res.Tem;
       
    catch ME 
        warning('this folder"%s" failed: %s', folders(k).name, ME.message);
    end
end

%plot svds:
T  = temps + 273.15;

figure;
for i = 1:length(vfits)
    plot(T, vfits{i});
    hold on;
end
hold off;
title('SVD fits component 2');

figure;
scatter(concs, Tms, 100, "filled");
title('TMs');

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
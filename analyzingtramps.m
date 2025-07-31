clear
close all
clc;  


disp('Starting…');
base = "C:\Users\ibend\data\PNIPAM_300kda_20_1_60_T_Ramps";
folders = dir(base);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name},{'.','..'}));

allres = cell(numel(folders),1);

Tms  = nan(numel(folders),1);
dTs  = nan(numel(folders),1);
concs = nan(numel(folders),1);
addpath("C:\Users\ibend\igor.lab");

for k = 1:numel(folders)
    fpath = fullfile(base, folders(k).name);
    fprintf('Processing: %s\n', fpath);
    temps = 20:60;
    concs(k) = parseConcFromName(folders(k).name);
    try
        analyze_tramp_folder(fpath, temps, concs(k));
    catch ME 
        warning('you dumbass this folder"%s" failed: %s', folders(k).name, ME.message);
    end
end

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
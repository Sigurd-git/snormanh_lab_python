function map = plot_circle_map2(subjid, values, hemi,out_path, varargin)

% Plots a heatmap of electrode values for one subject
%
% -- Required inputs --
%
% subjid: 'URXX'
% values: vector of values, one per electrode for that subject
% hemi: the hemisphere to plot

% supporting files in shared directory


shared_directory = '/scratch/snormanh_lab/shared/';
addpath(genpath([shared_directory '/code/lab-fmri-code'])); % need plotting code from this directory
addpath(genpath([shared_directory '/code/lab-analysis-code'])); % need plotting code from this directory
addpath(genpath([shared_directory '/code/lab-intracranial-code/'])); % need plotting code from this directory
addpath(genpath([shared_directory '/code/export_fig_v3']));

clear I;
I.cmap = 'cbrewer-blue-red';
I.range = [-1,1]*max(values); % range of the color map
I.figh = matlab.ui.Figure.empty;
I.plot = true; % enables you to suppress plotting and only return the map
I = parse_optInputs_keyvalue(varargin, I);

% load electrode coordinates
freesurfer_directory = [shared_directory '/electrode-localization/analysis/electrode-coords'];
coord_file = [freesurfer_directory '/' subjid '.mat'];
load(coord_file, 'E');
assert(length(E.verts)==length(values));

% check if there are any electrodes for this hemisphere
if ~any(ismember(E.hemi(~isnan(E.verts) & ~isnan(values)), hemi))
    fprintf('No electrodes for %s\n', hemi);
    map = [];
    return;
end

% convert a colormap name to a matrix of values
if ischar(I.cmap)
    I.cmap = cmap_from_name(I.cmap);
end
assert(ismatrix(I.cmap));

% create the map
map = circle_map_from_coord(E, values, subjid, hemi, 'brain', 'fsaverage');

% create figure handle if it doesn't exist
% good to have this be a standard size
if isempty(I.figh);
    I.figh = figure;
    pos = get(I.figh,'Position');
    set(I.figh, 'Position', [pos(1:2), 800 800]);
end

% plot the map
if I.plot
    plot_fsaverage_1D_overlay_v2(map, hemi, 'colormap', I.cmap, 'color_range', I.range, 'figh', I.figh);
    % export_fig(out_path, '-png','-r100','-nocrop');
    set(I.figh, 'PaperPosition', [4 4 5 5]);
    saveas(I.figh, out_path, 'png');
end


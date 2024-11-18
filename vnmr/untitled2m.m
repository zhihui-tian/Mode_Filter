clear;
clc;
close all;

% Load distinguishable colors
ColorSet = distinguishable_colors(100);

% Configure figure and axes for Figure 9
figure9 = figure(9);
axes9 = axes('Parent', figure9, ...
    'Box', 'on', ...
    'TickDir', 'in', ...
    'TickLength', [0.02 0.02], ...
    'XMinorTick', 'off', ...
    'YMinorTick', 'off', ...
    'YGrid', 'off', ...
    'XColor', 'k', ...
    'YColor', 'k', ...
    'XTick', 2:2:14, ...
    'XLim', [2 14], ...
    'YTick', -10:5:10, ...
    'YLim', [-10 10], ...
    'FontWeight', 'bold', ...
    'FontSize', 14, ...
    'LineWidth', 1.5);
hold on;

% Load data
timeStep = 100;
load("C:\Users\zhihui.tian\Desktop\MF_new\MF-main\8samples\GrainAreaAvg.mat");
load("C:\Users\zhihui.tian\Desktop\MF_new\MF-main\8samples\GrainAllArea.mat");
load("C:\Users\zhihui.tian\Desktop\MF_new\MF-main\8samples\GrainAllSides.mat");

% Ensure valid timeStep
if timeStep <= 1 || timeStep >= size(GrainAllArea, 2)
    error('Invalid timeStep: Ensure it is between 2 and %d.', size(GrainAllArea, 2) - 1);
end

GrainAreaAvg1 = GrainAreaAvg.';
GrainAllArea = GrainAllArea.';
GrainAllSides = GrainAllSides.';

PP = polyfit(1:100, GrainAreaAvg1(1:100,1), 1);
slope = PP(1);

% Process areas and sides
A1 = double(GrainAllArea(:, timeStep));
A2 = double(GrainAllArea(:, timeStep - 1));
A3 = double(GrainAllArea(:, timeStep + 1));
A = A1(A1 > 0);
dA_dt_norm = (A3(A1 > 0) - A2(A1 > 0)) / 2 / (pi / 3) / slope;

F = double(GrainAllSides(:, timeStep));
F = F(A1 > 0);

% Error bar calculation
[Favg, ~, idx] = unique(F);
dA_dt_normavg = accumarray(idx, dA_dt_norm, [], @mean);
dA_dt_normStd = accumarray(idx, dA_dt_norm, [], @std);

% Plot Figure 9
errorbar(Favg, dA_dt_normavg, dA_dt_normStd, '-s', 'MarkerSize', 10, 'LineWidth', 2);
plot(Favg, -(6 - Favg), 'k-', 'LineWidth', 2);
xlabel('Number of sides F', 'FontWeight', 'bold', 'FontSize', 18);
ylabel('{dA}/{dt}/({\pi/3} {\mu} {\sigma})', 'FontWeight', 'bold', 'FontSize', 18);
grid on;

% Figure 1: Scatter Plot
Rnorm = sqrt(A) / pi / mean(sqrt(A) / pi);
figure(1);
gscatter(Rnorm, dA_dt_norm, F, ColorSet, [], [], 'off');
xlabel('Normalized radius R/<R>', 'FontWeight', 'bold', 'FontSize', 18);
ylabel('{dA}/{dt}/({\pi/3} {\mu} {\sigma})', 'FontWeight', 'bold', 'FontSize', 18);

% Save Figures
export_fig(['MFMode_cov50_uniform_vNMR_timeStep' num2str(timeStep) '.png'], '-transparent', '-r600');
export_fig(['MFModeModel_cov50_uniform_dAdtvsNormradius_timeStep' num2str(timeStep) '.png'], '-transparent', '-r600');

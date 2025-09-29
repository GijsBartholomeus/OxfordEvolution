
% Run Budding Yeast Cell Cycle Model
% This script can be executed in VS Code with MATLAB extension

% Clear workspace
clear all; clc;

% Set the working directory to where our model is located
cd('/Users/gijsbartholomeus/Documents/STUDIE/OxfordEvolution/Code');

% Run the model
timespan = [0, 400];  % 400 time units
[T, Y, yinit, param, allNames, allValues] = BuddingYeastCellCycle_2015(timespan);

% Display basic info
fprintf('Simulation completed successfully!\n');
fprintf('Time span: %.2f to %.2f\n', T(1), T(end));
fprintf('Number of time points: %d\n', length(T));
fprintf('Number of species: %d\n', size(Y, 2));
fprintf('Number of total variables: %d\n', length(allNames));

% Save results for Python to read
save('matlab_results.mat', 'T', 'Y', 'yinit', 'param', 'allNames', 'allValues');

% Save data in CSV format for easy reading
writematrix(T, 'time_data.csv');
writematrix(Y, 'species_concentrations.csv');

% Save species names
fid = fopen('species_names.txt', 'w');
for i = 1:length(allNames)
    fprintf(fid, '%s\n', allNames{i});
end
fclose(fid);

% Plot some key species if desired
figure;
hold on;

% Find indices for key species
cln3_idx = find(strcmp(allNames, 'CLN3'), 1);
cln2_idx = find(strcmp(allNames, 'CLN2'), 1);
clb2t_idx = find(strcmp(allNames, 'CLB2T'), 1);
mass_idx = find(strcmp(allNames, 'MASS'), 1);

if ~isempty(cln3_idx)
    plot(T, allValues(:, cln3_idx), 'b-', 'LineWidth', 2, 'DisplayName', 'CLN3');
end
if ~isempty(cln2_idx)
    plot(T, allValues(:, cln2_idx), 'r-', 'LineWidth', 2, 'DisplayName', 'CLN2');
end
if ~isempty(clb2t_idx)
    plot(T, allValues(:, clb2t_idx), 'g-', 'LineWidth', 2, 'DisplayName', 'CLB2T');
end
if ~isempty(mass_idx)
    plot(T, allValues(:, mass_idx), 'm-', 'LineWidth', 2, 'DisplayName', 'MASS');
end

xlabel('Time');
ylabel('Concentration');
title('Kraikivski 2015 Budding Yeast Cell Cycle Model');
legend('show');
grid on;

fprintf('Results saved to:\n');
fprintf('  - matlab_results.mat (MATLAB format)\n');
fprintf('  - time_data.csv (time points)\n');
fprintf('  - species_concentrations.csv (concentration data)\n');
fprintf('  - species_names.txt (variable names)\n');

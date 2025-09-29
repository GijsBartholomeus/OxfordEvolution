
% Run Budding Yeast Cell Cycle Model
% This script can be executed in VS Code with MATLAB extension

% Clear workspace
clear all; clc;

% Set the working directory to where our model is located
cd('/Users/gijsbartholomeus/Documents/STUDIE/OxfordEvolution/Code');

% Run the model with extended time for multiple cycles
timespan = [0, 300];  % 300 time units to see ~2-3 cell cycles
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

% ========================================================================
% FOCUSED CELL CYCLE PLOTS - 3 KEY SPECIES
% ========================================================================

% Find indices for the most interesting cell cycle regulators
fprintf('\nFinding key species indices...\n');

% Three most informative species:
% 1. CLN2 - G1/S cyclin (early cell cycle activation)
% 2. CLB2 - G2/M cyclin (mitotic progression) 
% 3. SIC1 - CDK inhibitor (regulation/oscillation control)
cln2_idx = find(strcmp(allNames, 'CLN2'), 1);
clb2_idx = find(strcmp(allNames, 'CLB2'), 1);
sic1_idx = find(strcmp(allNames, 'SIC1'), 1);

% Print found indices
species_found = {};
if ~isempty(cln2_idx), species_found{end+1} = 'CLN2'; end
if ~isempty(clb2_idx), species_found{end+1} = 'CLB2'; end
if ~isempty(sic1_idx), species_found{end+1} = 'SIC1'; end

fprintf('Found %d key species: %s\n', length(species_found), strjoin(species_found, ', '));

% Create focused figure with 3 subplots
figure('Position', [100, 100, 1200, 800], 'Name', 'Key Cell Cycle Regulators');

% ---- Subplot 1: Time Series of All Three Species ----
subplot(2, 2, 1);
hold on;
if ~isempty(cln2_idx)
    plot(T, allValues(:, cln2_idx), 'r-', 'LineWidth', 3, 'DisplayName', 'CLN2 (G1/S Cyclin)');
end
if ~isempty(clb2_idx)
    plot(T, allValues(:, clb2_idx), 'b-', 'LineWidth', 3, 'DisplayName', 'CLB2 (G2/M Cyclin)');
end
if ~isempty(sic1_idx)
    plot(T, allValues(:, sic1_idx), 'k--', 'LineWidth', 3, 'DisplayName', 'SIC1 (CDK Inhibitor)');
end
xlabel('Time');
ylabel('Concentration');
title('Key Cell Cycle Regulators (0-300 time units)');
legend('show', 'Location', 'best');
grid on;
xlim([0, 300]);

% ---- Subplot 2: Normalized Comparison ----
subplot(2, 2, 2);
hold on;
if ~isempty(cln2_idx) && ~isempty(clb2_idx) && ~isempty(sic1_idx)
    % Normalize for comparison
    cln2_norm = allValues(:, cln2_idx) / max(allValues(:, cln2_idx));
    clb2_norm = allValues(:, clb2_idx) / max(allValues(:, clb2_idx));
    sic1_norm = allValues(:, sic1_idx) / max(allValues(:, sic1_idx));
    
    plot(T, cln2_norm, 'r-', 'LineWidth', 3, 'DisplayName', 'CLN2 (normalized)');
    plot(T, clb2_norm, 'b-', 'LineWidth', 3, 'DisplayName', 'CLB2 (normalized)');
    plot(T, sic1_norm, 'k--', 'LineWidth', 3, 'DisplayName', 'SIC1 (normalized)');
end
xlabel('Time');
ylabel('Normalized Concentration');
title('Normalized Cell Cycle Oscillations');
legend('show', 'Location', 'best');
grid on;
xlim([0, 300]);

% ---- Subplot 3: Phase Portrait (CLN2 vs CLB2) ----
subplot(2, 2, 3);
if ~isempty(cln2_idx) && ~isempty(clb2_idx)
    plot(allValues(:, cln2_idx), allValues(:, clb2_idx), 'b-', 'LineWidth', 2);
    xlabel('CLN2 (G1/S Cyclin)');
    ylabel('CLB2 (G2/M Cyclin)');
    title('Phase Portrait: CLN2 vs CLB2');
    grid on;
    
    % Mark start and end points
    hold on;
    plot(allValues(1, cln2_idx), allValues(1, clb2_idx), 'go', 'MarkerSize', 10, 'LineWidth', 3);
    plot(allValues(end, cln2_idx), allValues(end, clb2_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
    legend('Cell Cycle Trajectory', 'Start', 'End', 'Location', 'best');
end

% ---- Subplot 4: CLB2 vs SIC1 (Cyclin vs Inhibitor) ----
subplot(2, 2, 4);
if ~isempty(clb2_idx) && ~isempty(sic1_idx)
    plot(allValues(:, sic1_idx), allValues(:, clb2_idx), 'k-', 'LineWidth', 2);
    xlabel('SIC1 (CDK Inhibitor)');
    ylabel('CLB2 (G2/M Cyclin)');
    title('Inhibition Dynamics: SIC1 vs CLB2');
    grid on;
    
    % Mark start and end points
    hold on;
    plot(allValues(1, sic1_idx), allValues(1, clb2_idx), 'go', 'MarkerSize', 10, 'LineWidth', 3);
    plot(allValues(end, sic1_idx), allValues(end, clb2_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
    legend('Inhibition Trajectory', 'Start', 'End', 'Location', 'best');
end

% Add overall title
sgtitle('Budding Yeast Cell Cycle: Key Regulatory Dynamics', 'FontSize', 16, 'FontWeight', 'bold');

% Save the figure
saveas(gcf, 'yeast_cell_cycle_analysis.fig');
saveas(gcf, 'yeast_cell_cycle_analysis.png');

% ---- Additional: Phase Portrait Analysis ----
figure('Position', [150, 150, 800, 600], 'Name', 'Cell Cycle Phase Analysis');

% Create phase portrait: CLN2 vs CLB2 (G1/S vs G2/M)
if ~isempty(cln2_idx) && ~isempty(clb2_idx)
    subplot(2, 2, 1);
    plot(allValues(:, cln2_idx), allValues(:, clb2_idx), 'b-', 'LineWidth', 2);
    xlabel('CLN2 (G1/S Cyclin)');
    ylabel('CLB2 (G2/M Cyclin)');
    title('Phase Portrait: CLN2 vs CLB2');
    grid on;
    
    % Mark start and end points
    hold on;
    plot(allValues(1, cln2_idx), allValues(1, clb2_idx), 'go', 'MarkerSize', 8, 'LineWidth', 2);
    plot(allValues(end, cln2_idx), allValues(end, clb2_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    legend('Trajectory', 'Start', 'End', 'Location', 'best');
end

% CLB2 vs Cell Mass
if ~isempty(clb2_idx) && ~isempty(mass_idx)
    subplot(2, 2, 2);
    plot(allValues(:, mass_idx), allValues(:, clb2_idx), 'r-', 'LineWidth', 2);
    xlabel('Cell Mass');
    ylabel('CLB2 (Mitotic Cyclin)');
    title('CLB2 vs Cell Mass');
    grid on;
end

% SIC1 vs CLB2 (Inhibitor vs Cyclin)
if ~isempty(sic1_idx) && ~isempty(clb2_idx)
    subplot(2, 2, 3);
    plot(allValues(:, sic1_idx), allValues(:, clb2_idx), 'k-', 'LineWidth', 2);
    xlabel('SIC1 (CDK Inhibitor)');
    ylabel('CLB2 (G2/M Cyclin)');
    title('SIC1 vs CLB2: Inhibition Dynamics');
    grid on;
end

% Time series overlay of key oscillating species
if ~isempty(cln2_idx) && ~isempty(clb2_idx) && ~isempty(sic1_idx)
    subplot(2, 2, 4);
    hold on;
    
    % Normalize for comparison
    cln2_norm = allValues(:, cln2_idx) / max(allValues(:, cln2_idx));
    clb2_norm = allValues(:, clb2_idx) / max(allValues(:, clb2_idx));
    sic1_norm = allValues(:, sic1_idx) / max(allValues(:, sic1_idx));
    
    plot(T, cln2_norm, 'r-', 'LineWidth', 2, 'DisplayName', 'CLN2 (norm)');
    plot(T, clb2_norm, 'b-', 'LineWidth', 2, 'DisplayName', 'CLB2 (norm)');
    plot(T, sic1_norm, 'k--', 'LineWidth', 2, 'DisplayName', 'SIC1 (norm)');
    
    xlabel('Time');
    ylabel('Normalized Concentration');
    title('Normalized Oscillations');
    legend('show', 'Location', 'best');
    grid on;
    xlim([0, 300]);
end

saveas(gcf, 'yeast_phase_analysis.fig');
saveas(gcf, 'yeast_phase_analysis.png');

% ========================================================================
% ANALYSIS SUMMARY
% ========================================================================

fprintf('\n=== BUDDING YEAST CELL CYCLE ANALYSIS COMPLETE ===\n');
fprintf('Model: Kraikivski et al. 2015\n');
fprintf('Time span: %.1f time units (approximately %.1f cell cycles)\n', T(end), T(end)/120);

% Calculate cell cycle period (estimate from CLB2 oscillations)
if ~isempty(clb2_idx)
    clb2_values = allValues(:, clb2_idx);
    % Find peaks to estimate period
    [peaks, peak_locs] = findpeaks(clb2_values, 'MinPeakHeight', max(clb2_values)*0.5);
    if length(peak_locs) > 1
        periods = diff(T(peak_locs));
        avg_period = mean(periods);
        fprintf('Estimated cell cycle period: %.1f ± %.1f time units\n', avg_period, std(periods));
        fprintf('Number of complete cycles observed: %d\n', length(peaks));
    end
end

% Report key species concentrations at final time
fprintf('\nFinal concentrations of key regulators:\n');
final_time_idx = length(T);
key_species = {{'CLN3', cln3_idx}, {'CLN2', cln2_idx}, {'CLB5', clb5_idx}, ...
               {'CLB2', clb2_idx}, {'SIC1', sic1_idx}, {'CDC6', cdc6_idx}, ...
               {'MASS', mass_idx}};

for i = 1:length(key_species)
    name = key_species{i}{1};
    idx = key_species{i}{2};
    if ~isempty(idx)
        fprintf('  %s: %.3f\n', name, allValues(final_time_idx, idx));
    end
end

fprintf('\nFiles saved:\n');
fprintf('  - matlab_results.mat (MATLAB format)\n');
fprintf('  - time_data.csv (time points)\n');
fprintf('  - species_concentrations.csv (concentration data)\n');
fprintf('  - species_names.txt (variable names)\n');
fprintf('  - yeast_cell_cycle_analysis.png (main plots)\n');
fprintf('  - yeast_phase_analysis.png (phase portraits)\n');
fprintf('  - yeast_cell_cycle_analysis.fig (MATLAB figure)\n');
fprintf('  - yeast_phase_analysis.fig (MATLAB figure)\n');

fprintf('\n=== Analysis completed successfully! ===\n');

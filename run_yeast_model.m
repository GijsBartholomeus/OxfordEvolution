
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
% COMPREHENSIVE CELL CYCLE PLOTS
% ========================================================================

% Find indices for key cell cycle regulators
fprintf('\nFinding key species indices...\n');

% Core cyclins (Cln1+Cln2='Cln2', Clb5+Clb6='Clb5', Clb1+Clb2='Clb2')
cln3_idx = find(strcmp(allNames, 'CLN3'), 1);
cln2_idx = find(strcmp(allNames, 'CLN2'), 1);
clb5_idx = find(strcmp(allNames, 'CLB5'), 1);
clb2_idx = find(strcmp(allNames, 'CLB2'), 1);

% CDK inhibitors (CKI = Sic1 + Cdc6)
sic1_idx = find(strcmp(allNames, 'SIC1'), 1);
cdc6_idx = find(strcmp(allNames, 'CDC6'), 1);

% Cell cycle checkpoints and regulation
cdc20_idx = find(strcmp(allNames, 'CDC20A'), 1);
cdh1_idx = find(strcmp(allNames, 'CDH1A'), 1);
apc_idx = find(strcmp(allNames, 'APCP'), 1);

% Cell mass and size
mass_idx = find(strcmp(allNames, 'MASS'), 1);

% DNA replication and spindle
ori_idx = find(strcmp(allNames, 'ORI'), 1);
spindle_idx = find(strcmp(allNames, 'SPN'), 1);

% Transcription factors
swi4_idx = find(strcmp(allNames, 'SWI4'), 1);
swi6_idx = find(strcmp(allNames, 'SWI6'), 1);
mcm1_idx = find(strcmp(allNames, 'MCM1'), 1);

% Cell cycle phases markers
bud_idx = find(strcmp(allNames, 'BUD'), 1);

% Print found indices
species_found = {};
if ~isempty(cln3_idx), species_found{end+1} = 'CLN3'; end
if ~isempty(cln2_idx), species_found{end+1} = 'CLN2'; end
if ~isempty(clb5_idx), species_found{end+1} = 'CLB5'; end
if ~isempty(clb2_idx), species_found{end+1} = 'CLB2'; end
if ~isempty(sic1_idx), species_found{end+1} = 'SIC1'; end
if ~isempty(cdc6_idx), species_found{end+1} = 'CDC6'; end
if ~isempty(cdc20_idx), species_found{end+1} = 'CDC20A'; end
if ~isempty(cdh1_idx), species_found{end+1} = 'CDH1A'; end
if ~isempty(mass_idx), species_found{end+1} = 'MASS'; end

fprintf('Found %d key species: %s\n', length(species_found), strjoin(species_found, ', '));

% Create comprehensive figure with multiple subplots
figure('Position', [100, 100, 1400, 1000], 'Name', 'Budding Yeast Cell Cycle Analysis');

% ---- Subplot 1: Core Cyclins (G1/S/G2/M progression) ----
subplot(3, 2, 1);
hold on;
if ~isempty(cln3_idx)
    plot(T, allValues(:, cln3_idx), 'b-', 'LineWidth', 2.5, 'DisplayName', 'CLN3 (G1)');
end
if ~isempty(cln2_idx)
    plot(T, allValues(:, cln2_idx), 'r-', 'LineWidth', 2.5, 'DisplayName', 'CLN2 (G1/S)');
end
if ~isempty(clb5_idx)
    plot(T, allValues(:, clb5_idx), 'g-', 'LineWidth', 2.5, 'DisplayName', 'CLB5 (S)');
end
if ~isempty(clb2_idx)
    plot(T, allValues(:, clb2_idx), 'm-', 'LineWidth', 2.5, 'DisplayName', 'CLB2 (G2/M)');
end
xlabel('Time');
ylabel('Concentration');
title('Core Cyclins: Cell Cycle Progression');
legend('show', 'Location', 'best');
grid on;
xlim([0, 300]);

% ---- Subplot 2: CDK Inhibitors (CKI) ----
subplot(3, 2, 2);
hold on;
if ~isempty(sic1_idx)
    plot(T, allValues(:, sic1_idx), 'k-', 'LineWidth', 2.5, 'DisplayName', 'SIC1');
end
if ~isempty(cdc6_idx)
    plot(T, allValues(:, cdc6_idx), 'c-', 'LineWidth', 2.5, 'DisplayName', 'CDC6');
end
xlabel('Time');
ylabel('Concentration');
title('CDK Inhibitors (CKI): SIC1 & CDC6');
legend('show', 'Location', 'best');
grid on;
xlim([0, 300]);

% ---- Subplot 3: Cell Cycle Checkpoints ----
subplot(3, 2, 3);
hold on;
if ~isempty(cdc20_idx)
    plot(T, allValues(:, cdc20_idx), 'r-', 'LineWidth', 2.5, 'DisplayName', 'CDC20A');
end
if ~isempty(cdh1_idx)
    plot(T, allValues(:, cdh1_idx), 'b-', 'LineWidth', 2.5, 'DisplayName', 'CDH1A');
end
if ~isempty(apc_idx)
    plot(T, allValues(:, apc_idx), 'g-', 'LineWidth', 2.5, 'DisplayName', 'APC^P');
end
xlabel('Time');
ylabel('Concentration');
title('Cell Cycle Checkpoints: APC Regulation');
legend('show', 'Location', 'best');
grid on;
xlim([0, 300]);

% ---- Subplot 4: Cell Mass and Growth ----
subplot(3, 2, 4);
hold on;
if ~isempty(mass_idx)
    plot(T, allValues(:, mass_idx), 'k-', 'LineWidth', 3, 'DisplayName', 'Cell Mass');
end
if ~isempty(bud_idx)
    plot(T, allValues(:, bud_idx), 'orange', 'LineWidth', 2.5, 'DisplayName', 'Bud Formation');
end
xlabel('Time');
ylabel('Concentration');
title('Cell Growth and Morphology');
legend('show', 'Location', 'best');
grid on;
xlim([0, 300]);

% ---- Subplot 5: DNA Replication ----
subplot(3, 2, 5);
hold on;
if ~isempty(ori_idx)
    plot(T, allValues(:, ori_idx), 'purple', 'LineWidth', 2.5, 'DisplayName', 'ORI (DNA Replication)');
end
if ~isempty(spindle_idx)
    plot(T, allValues(:, spindle_idx), 'brown', 'LineWidth', 2.5, 'DisplayName', 'Spindle');
end
xlabel('Time');
ylabel('Concentration');
title('DNA Replication and Spindle Formation');
legend('show', 'Location', 'best');
grid on;
xlim([0, 300]);

% ---- Subplot 6: Transcriptional Regulation ----
subplot(3, 2, 6);
hold on;
if ~isempty(swi4_idx)
    plot(T, allValues(:, swi4_idx), 'cyan', 'LineWidth', 2.5, 'DisplayName', 'SWI4');
end
if ~isempty(swi6_idx)
    plot(T, allValues(:, swi6_idx), 'magenta', 'LineWidth', 2.5, 'DisplayName', 'SWI6');
end
if ~isempty(mcm1_idx)
    plot(T, allValues(:, mcm1_idx), 'yellow', 'LineWidth', 2.5, 'DisplayName', 'MCM1');
end
xlabel('Time');
ylabel('Concentration');
title('Transcriptional Regulators');
legend('show', 'Location', 'best');
grid on;
xlim([0, 300]);

% Add overall title
sgtitle('Budding Yeast Cell Cycle Regulation (Kraikivski 2015 Model)', 'FontSize', 16, 'FontWeight', 'bold');

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

fprintf('Results saved to:\n');
fprintf('  - matlab_results.mat (MATLAB format)\n');
fprintf('  - time_data.csv (time points)\n');
fprintf('  - species_concentrations.csv (concentration data)\n');
fprintf('  - species_names.txt (variable names)\n');

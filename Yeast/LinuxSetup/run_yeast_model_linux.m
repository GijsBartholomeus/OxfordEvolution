% LINUX-OPTIMIZED YEAST CELL CYCLE ANALYSIS
% Clean, minimal version for copy-paste
% Compatible with both MATLAB and Octave

clear all; clc;

% Setup
fprintf('Starting Yeast Cell Cycle Analysis...\n');
timespan = [0, 300];

% Run simulation
[T, Y, yinit, param, allNames, allValues] = BuddingYeastCellCycle_2015(timespan);
fprintf('Simulation complete: %d time points, %d species\n', length(T), length(allNames));

% Find key species (with error handling)
cln2_idx = find(strcmp(allNames, 'CLN2'), 1);
clb2_idx = find(strcmp(allNames, 'CLB2'), 1);
ckit_idx = find(strcmp(allNames, 'CKIT'), 1);

if isempty(cln2_idx), cln2_idx = find(strcmp(allNames, 'CLN3'), 1); end
if isempty(ckit_idx), ckit_idx = find(strcmp(allNames, 'CKIP'), 1); end

fprintf('Found species: CLN2=%d, CLB2=%d, CKIT=%d\n', ...
        ~isempty(cln2_idx), ~isempty(clb2_idx), ~isempty(ckit_idx));

% Save data
save('matlab_results.mat', 'T', 'Y', 'allNames', 'allValues', '-v7');
csvwrite('time_data.csv', T);
csvwrite('species_data.csv', allValues);

% Create plot (Linux-compatible)
figure('Visible', 'off');
set(gcf, 'Position', [100, 100, 1000, 600]);

% Main time series plot
subplot(2, 2, 1);
hold on;
if ~isempty(cln2_idx)
    plot(T, allValues(:, cln2_idx), 'r-', 'LineWidth', 2);
end
if ~isempty(clb2_idx)
    plot(T, allValues(:, clb2_idx), 'b-', 'LineWidth', 2);
end
if ~isempty(ckit_idx)
    plot(T, allValues(:, ckit_idx), 'k--', 'LineWidth', 2);
end
xlabel('Time (arbitrary units)');
ylabel('Concentration');
title('Key Cell Cycle Regulators');
legend('CLN2 (G1/S)', 'CLB2 (G2/M)', 'CKIT (Inhibitor)', 'Location', 'best');
grid on;
xlim([0, 300]);

% Phase portrait
subplot(2, 2, 2);
if ~isempty(cln2_idx) && ~isempty(clb2_idx)
    plot(allValues(:, cln2_idx), allValues(:, clb2_idx), 'b-', 'LineWidth', 1.5);
    xlabel('CLN2 (G1/S Cyclin)');
    ylabel('CLB2 (G2/M Cyclin)');
    title('Phase Portrait: CLN2 vs CLB2');
    grid on;
end

% Normalized oscillations
subplot(2, 2, 3);
if ~isempty(cln2_idx) && ~isempty(clb2_idx) && ~isempty(ckit_idx)
    cln2_norm = allValues(:, cln2_idx) / max(allValues(:, cln2_idx));
    clb2_norm = allValues(:, clb2_idx) / max(allValues(:, clb2_idx));
    ckit_norm = allValues(:, ckit_idx) / max(allValues(:, ckit_idx));
    
    hold on;
    plot(T, cln2_norm, 'r-', 'LineWidth', 1.5);
    plot(T, clb2_norm, 'b-', 'LineWidth', 1.5);
    plot(T, ckit_norm, 'k--', 'LineWidth', 1.5);
    xlabel('Time');
    ylabel('Normalized Concentration');
    title('Normalized Oscillations');
    legend('CLN2', 'CLB2', 'CKIT', 'Location', 'best');
    grid on;
    xlim([0, 300]);
end

% Period analysis
subplot(2, 2, 4);
if ~isempty(clb2_idx)
    clb2_signal = allValues(:, clb2_idx);
    plot(T, clb2_signal, 'b-', 'LineWidth', 2);
    xlabel('Time');
    ylabel('CLB2 Concentration');
    title('CLB2 Oscillations (Period Analysis)');
    grid on;
    
    % Simple period estimation
    clb2_max = max(clb2_signal);
    clb2_min = min(clb2_signal);
    threshold = clb2_min + 0.7 * (clb2_max - clb2_min);
    hold on;
    plot([T(1), T(end)], [threshold, threshold], 'r--', 'LineWidth', 1);
    
    % Count cycles
    above_threshold = clb2_signal > threshold;
    transitions = diff(above_threshold);
    rising_edges = find(transitions == 1);
    
    if length(rising_edges) > 1
        periods = diff(T(rising_edges));
        avg_period = mean(periods);
        fprintf('Estimated period: %.1f ± %.1f time units\n', avg_period, std(periods));
        fprintf('Number of cycles: %d\n', length(rising_edges));
    end
end

% Save figure
print('-dpng', '-r150', 'yeast_analysis_linux.png');
close(gcf);

% Summary
fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Files generated:\n');
fprintf('  - matlab_results.mat (simulation data)\n');
fprintf('  - time_data.csv (time points)\n');
fprintf('  - species_data.csv (all species concentrations)\n');
fprintf('  - yeast_analysis_linux.png (plots)\n');

if ~isempty(clb2_idx)
    fprintf('\nFinal concentrations:\n');
    if ~isempty(cln2_idx), fprintf('  CLN2: %.3f\n', allValues(end, cln2_idx)); end
    if ~isempty(clb2_idx), fprintf('  CLB2: %.3f\n', allValues(end, clb2_idx)); end
    if ~isempty(ckit_idx), fprintf('  CKIT: %.3f\n', allValues(end, ckit_idx)); end
end

fprintf('\nAnalysis completed successfully!\n');
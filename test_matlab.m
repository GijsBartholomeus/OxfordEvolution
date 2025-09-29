% Simple MATLAB Test Script
% This script tests basic MATLAB functionality and file I/O

% Clear workspace
clear all; clc;

fprintf('=== MATLAB Test Script ===\n');
fprintf('MATLAB Version: %s\n', version);
fprintf('Current directory: %s\n', pwd);
fprintf('Current time: %s\n', datestr(now));

% Test basic mathematics
fprintf('\n--- Basic Math Test ---\n');
a = 5;
b = 3;
c = a + b;
fprintf('Testing: %d + %d = %d\n', a, b, c);

% Test vector operations
fprintf('\n--- Vector Operations Test ---\n');
x = [1, 2, 3, 4, 5];
y = x.^2;
fprintf('x = [1, 2, 3, 4, 5]\n');
fprintf('x^2 = [%s]\n', sprintf('%.0f ', y));

% Test plotting capability
fprintf('\n--- Plotting Test ---\n');
try
    figure('Visible', 'off');  % Create figure without displaying
    plot(x, y, 'b-o', 'LineWidth', 2);
    xlabel('x');
    ylabel('x^2');
    title('Simple Test Plot');
    grid on;
    
    % Save the plot
    saveas(gcf, 'test_plot.png');
    close(gcf);
    fprintf('Plot created and saved successfully as test_plot.png\n');
catch ME
    fprintf('Plotting failed: %s\n', ME.message);
end

% Test file I/O
fprintf('\n--- File I/O Test ---\n');
try
    % Write test data
    test_data = [1:10; (1:10).^2];
    csvwrite('test_data.csv', test_data);
    fprintf('Test data written to test_data.csv\n');
    
    % Read test data back
    read_data = csvread('test_data.csv');
    fprintf('Test data read back successfully\n');
    fprintf('Data size: %dx%d\n', size(read_data, 1), size(read_data, 2));
catch ME
    fprintf('File I/O failed: %s\n', ME.message);
end

% Test current working directory and file listing
fprintf('\n--- Directory Test ---\n');
files = dir('*.m');
fprintf('MATLAB files in current directory:\n');
for i = 1:length(files)
    fprintf('  %s\n', files(i).name);
end

% Test if key files exist
fprintf('\n--- Key Files Check ---\n');
key_files = {'BuddingYeastCellCycle_2015.m', 'run_yeast_model.m'};
for i = 1:length(key_files)
    if exist(key_files{i}, 'file')
        fprintf('✓ %s exists\n', key_files{i});
    else
        fprintf('✗ %s NOT FOUND\n', key_files{i});
    end
end

% Memory and performance info
fprintf('\n--- System Info ---\n');
try
    mem_info = memory;
    fprintf('Available memory: %.2f GB\n', mem_info.MemAvailableAllArrays / 1024^3);
catch
    fprintf('Memory info not available on this system\n');
end

fprintf('\n=== MATLAB Test Completed Successfully ===\n');
fprintf('Test completed at: %s\n', datestr(now));
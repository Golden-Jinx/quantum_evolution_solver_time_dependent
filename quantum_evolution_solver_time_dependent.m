%% General Time-Dependent Schrödinger Equation Solver

clear; close all; clc;

%% ==================== Parameter Settings ====================
% Spatial parameters
params.x_range = [-20, 20];        % Spatial range
params.Nx = 512;                   % Number of spatial grid points

% Time parameters
params.t_range = [0, 20];          % Time range (20 seconds)
params.Nt = 1200;                  % Number of time steps (20s*60fps=1200 frames)

% Physical constants (using atomic units)
params.hbar = 1;                   % Reduced Planck constant
params.m = 1;                      % Particle mass

%% ==================== Custom Function Area ====================
% Modify your potential function and initial wavefunction here

% ===== Potential Function Definition =====
% Example 1: Oscillating harmonic oscillator
% A = 0.3; omega = 1.5;
% V = @(x,t) 0.5*x.^2.*(1 + A*sin(omega*t));
% V_str = '0.5x^2[1+0.3sin(1.5t)]';

% Example 2: Complex dissipative potential
V = @(x,t) 0.5*x.^2 - 0.05i*exp(-x.^2/4);
V_str = '0.5x^2 - 0.05i e^{-x^2/4}';

% Example 3: Double potential well
% V = @(x,t) 0.25*(x.^2 - 4).^2;
% V_str = '0.25(x^2-4)^2';

% Example 4: Free particle
% V = @(x,t) 0*x;
% V_str = '0 (Free particle)';

% ===== Initial Wavefunction: Gaussian Wave Packet =====
x0 = -5;       % Initial position
p0 = 3;        % Initial momentum
sigma0 = 1.5;  % Wave packet width
psi0 = @(x) (1/(2*pi*sigma0^2))^(1/4) * ...
            exp(-(x-x0).^2/(4*sigma0^2) + 1i*p0*x);

% Initial wavefunction description string
psi0_str = sprintf('Gaussian wave packet: x_0=%.1f, p_0=%.1f, σ_0=%.1f', x0, p0, sigma0);

%% ==================== Grid Initialization ====================
fprintf('Initializing computational grid...\n');

x = linspace(params.x_range(1), params.x_range(2), params.Nx).';
dx = x(2) - x(1);

dk = 2*pi/(params.Nx*dx);
k = [0:params.Nx/2-1, -params.Nx/2:-1].' * dk;

t = linspace(params.t_range(1), params.t_range(2), params.Nt);
dt = t(2) - t(1);

%% ==================== Initialization and Precomputation ====================
% Initialize wavefunction and normalize
psi = psi0(x);
psi = psi / sqrt(trapz(x, abs(psi).^2));  % Normalization

fprintf('\n=== Initialization Info ===\n');
fprintf('Potential function: V(x,t) = %s\n', V_str);
fprintf('Initial wavefunction: %s\n', psi0_str);
fprintf('Evolution time: %.1f seconds\n', params.t_range(2));
fprintf('Total frames: %d\n', params.Nt);
fprintf('Initial normalization: ∫|ψ|^2dx = %.8f\n', trapz(x, abs(psi).^2));

% Detect potential type
V_test = V(x(1), t(1));
if ~isreal(V_test)
    fprintf('Complex potential detected: Imaginary part indicates dissipation/gain\n');
end

% Kinetic energy operator
T_k = params.hbar^2 * k.^2 / (2*params.m);
expT = exp(-1i * T_k * dt / params.hbar);

% Preallocate memory
fprintf('Preallocating memory...\n');
psi_history = zeros(params.Nx, params.Nt, 'like', 1i);
prob_history = zeros(params.Nx, params.Nt);
V_history = zeros(params.Nx, params.Nt, 'like', 1i);
norm_history = zeros(1, params.Nt);

k_plot = fftshift(k);
phi_k_history = zeros(length(k_plot), params.Nt);

%% ==================== Step 1: Compute All Time Steps ====================
fprintf('\n========== Step 1: Calculating Evolution Process ==========\n');

psi_history(:,1) = psi;
prob_history(:,1) = abs(psi).^2;
V_history(:,1) = V(x, t(1));
norm_history(1) = trapz(x, prob_history(:,1));

psi_k = fft(psi);
phi_k_history(:,1) = fftshift(abs(psi_k).^2);

tic;
for n = 1:params.Nt-1
    current_t = t(n);
    V_half = V(x, current_t + dt/2);
    
    % Split-operator method
    expV_half = exp(-1i * V_half * dt / (2*params.hbar));
    psi = expV_half .* psi;
    psi_k = fft(psi);
    psi_k = expT .* psi_k;
    psi = ifft(psi_k);
    psi = expV_half .* psi;
    
    psi_history(:,n+1) = psi;
    prob_history(:,n+1) = abs(psi).^2;
    V_history(:,n+1) = V(x, t(n+1));
    norm_history(n+1) = trapz(x, prob_history(:,n+1));
    
    phi_k_history(:,n+1) = fftshift(abs(psi_k).^2);
    
    if mod(n, floor(params.Nt/10)) == 0
        elapsed = toc;
        fprintf('Calculation progress: %.0f%% (Time elapsed: %.1fs)\n', 100*n/params.Nt, elapsed);
    end
end
calc_time = toc;
fprintf('Calculation complete! Total time: %.2fs\n', calc_time);

%% ==================== Determine Axis Ranges ====================
V_real = real(V_history(:));
V_min = min(V_real);
V_max = max(V_real);
V_range = V_max - V_min;
if V_range == 0
    V_range = 1;
end
V_ylim = [V_min - 0.1*V_range, V_max + 0.1*V_range];

prob_max = max(prob_history(:));
prob_ylim = [0, prob_max * 1.1];

psi_max = max(abs(psi_history(:)));
psi_ylim = [-psi_max * 1.2, psi_max * 1.2];

phi_k_max = max(phi_k_history(:));
phi_k_ylim = [0, phi_k_max * 1.1];
k_range = [-max(k)/3, max(k)/3];

%% ==================== Step 2: Play Animation ====================
fprintf('\n========== Step 2: Playing Animation (60 FPS) ==========\n');
fprintf('Calculation finished, preparing to play animation...\n');
fprintf('Press any key to start playback...\n');
pause;

fig = figure('Position', [150, 150, 1200, 700], ...
             'Renderer', 'opengl', ...
             'DoubleBuffer', 'on', ...
             'Name', 'Schrödinger Equation Evolution', ...
             'NumberTitle', 'off');

% Subplot 1: Probability density and potential
subplot(2,2,1);
yyaxis left;
h_prob = plot(x, prob_history(:,1), 'b-', 'LineWidth', 2);
ylabel('Probability density |\psi|^2', 'FontSize', 12);
ylim(prob_ylim);

yyaxis right;
h_V = plot(x, real(V_history(:,1)), 'r-', 'LineWidth', 2);
ylabel('Potential Re[V(x,t)]', 'FontSize', 12);
ylim(V_ylim);

xlabel('Position x', 'FontSize', 12);
h_title1 = title(sprintf('t = %.2f s', t(1)), 'FontSize', 12);
legend([h_prob, h_V], {'|\psi|^2', 'Re[V(x,t)]'}, 'Location', 'best');
grid on;
xlim([x(1), x(end)]);

% Subplot 2: Wavefunction
subplot(2,2,2);
h_real = plot(x, real(psi_history(:,1)), 'g-', 'LineWidth', 1.5);
hold on;
h_imag = plot(x, imag(psi_history(:,1)), 'm-', 'LineWidth', 1.5);
h_abs = plot(x, abs(psi_history(:,1)), 'k--', 'LineWidth', 1);
xlabel('Position x', 'FontSize', 12);
ylabel('Wavefunction \psi', 'FontSize', 12);
h_title2 = title(sprintf('t = %.2f s', t(1)), 'FontSize', 12);
legend([h_real, h_imag, h_abs], {'Re(\psi)', 'Im(\psi)', '|\psi|'}, 'Location', 'best');
grid on;
xlim([x(1), x(end)]);
ylim(psi_ylim);

% Subplot 3: Momentum-space wavefunction φ(k)
subplot(2,2,3);
h_phi_k = plot(k_plot, phi_k_history(:,1), 'b-', 'LineWidth', 2);
xlabel('Momentum k', 'FontSize', 12);
ylabel('|\phi(k)|^2', 'FontSize', 12);
h_title3 = title(sprintf('t = %.2f s', t(1)), 'FontSize', 12);
grid on;
xlim(k_range);
ylim(phi_k_ylim);

% Subplot 4: Probability conservation check
subplot(2,2,4);
h_norm = plot(t(1), norm_history(1), 'b-', 'LineWidth', 2);
hold on;
yline(norm_history(1), 'r--', 'LineWidth', 1);
xlabel('Time t (s)', 'FontSize', 12);
ylabel('Total probability ∫|ψ|^2dx', 'FontSize', 12);
h_title4 = title('Probability Evolution', 'FontSize', 12);
legend(h_norm, {'Total probability'}, 'Location', 'best');
grid on;
xlim([t(1), t(end)]);
norm_ylim = [min(norm_history)*0.99, max(norm_history)*1.01];
if norm_ylim(2) - norm_ylim(1) < 0.01
    norm_ylim = [norm_history(1)-0.1, norm_history(1)+0.1];
end
ylim(norm_ylim);

% Main title - auto-updates based on potential function
sgtitle(sprintf('Time-Dependent Schrödinger Equation Evolution | V(x,t) = %s', V_str), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'tex');

% Playback
target_fps = 60;
target_frame_time = 1/target_fps;
play_tic = tic;

fprintf('\nStarting animation playback...\n');

for n = 1:params.Nt
    % Update data
    set(h_prob, 'YData', prob_history(:,n));
    set(h_V, 'YData', real(V_history(:,n)));
    set(h_real, 'YData', real(psi_history(:,n)));
    set(h_imag, 'YData', imag(psi_history(:,n)));
    set(h_abs, 'YData', abs(psi_history(:,n)));
    set(h_phi_k, 'YData', phi_k_history(:,n));
    
    set(h_norm, 'XData', t(1:n), 'YData', norm_history(1:n));
    
    % Update titles
    set(h_title1, 'String', sprintf('t = %.2f s', t(n)));
    set(h_title2, 'String', sprintf('t = %.2f s', t(n)));
    set(h_title3, 'String', sprintf('t = %.2f s', t(n)));
    
    % Control frame rate
    if n < params.Nt
        elapsed = toc(play_tic);
        pause_time = n * target_frame_time - elapsed;
        if pause_time > 0
            pause(pause_time);
        end
    end
    drawnow;
end

play_time = toc(play_tic);
fprintf('\nPlayback complete! Actual frame rate: %.1f FPS\n', params.Nt/play_time);

%% ==================== Step 3: 3D Visualization ====================
fprintf('\n========== Step 3: Generating 3D Visualization ==========\n');

fig3d = figure('Position', [200, 200, 1000, 400], ...
               'Name', '3D Visualization', ...
               'NumberTitle', 'off');

% Probability density evolution
subplot(1,2,1);
[X_grid, T_grid] = meshgrid(x, t);
surf(X_grid, T_grid, prob_history.', 'EdgeColor', 'none');
colormap(gca, 'jet');
colorbar;
xlabel('Position x', 'FontSize', 12);
ylabel('Time t (s)', 'FontSize', 12);
zlabel('Probability density |\psi|^2', 'FontSize', 12);
title(sprintf('Wavefunction Spacetime Evolution (%.1fs)', params.t_range(2)), 'FontSize', 14);
view(45, 30);
grid on;

% Real part of potential
subplot(1,2,2);
surf(X_grid, T_grid, real(V_history).', 'EdgeColor', 'none');
colormap(gca, 'jet');
colorbar;
xlabel('Position x', 'FontSize', 12);
ylabel('Time t (s)', 'FontSize', 12);
zlabel('Potential Re[V(x,t)]', 'FontSize', 12);
title(sprintf('Potential Spacetime Distribution (%.1fs)', params.t_range(2)), 'FontSize', 14);
view(45, 30);
grid on;

% 3D figure main title - auto-updates based on potential function
sgtitle(sprintf('Numerical Solution of Time-Dependent Schrödinger Equation - 3D View | V(x,t) = %s', V_str), ...
        'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'tex');

fprintf('3D plot generated.\n');

%% ==================== Step 4: Save Video (Optional) ====================
fprintf('\n========== Step 4: Save Video ==========\n');

% Use try-catch for input handling
try
    save_video = input('Save video? (1=Yes, 0=No): ');
catch
    save_video = 0;
    fprintf('Invalid input, skipping video save.\n');
end

if save_video == 1
    video_filename = sprintf('quantum_evolution_t=%.1fs', params.t_range(2));
    
    % Check MPEG-4 format support
    try
        v = VideoWriter(video_filename, 'MPEG-4');
    catch
        v = VideoWriter(video_filename);  % Use default format
    end
    
    v.FrameRate = 60;
    v.Quality = 95;
    open(v);
    
    fprintf('Generating video: %s\n', [video_filename '.mp4']);
    
    % Create new figure window with fixed size
    fig2 = figure('Position', [150, 150, 1200, 700], ...
                  'Renderer', 'opengl');
    
    subplot(2,2,1);
    yyaxis left;
    h_prob2 = plot(x, prob_history(:,1), 'b-', 'LineWidth', 2);
    ylabel('Probability density |\psi|^2', 'FontSize', 12);
    ylim(prob_ylim);
    
    yyaxis right;
    h_V2 = plot(x, real(V_history(:,1)), 'r-', 'LineWidth', 2);
    ylabel('Potential Re[V(x,t)]', 'FontSize', 12);
    ylim(V_ylim);
    
    xlabel('Position x', 'FontSize', 12);
    h_title1_2 = title(sprintf('t = %.2f s', t(1)), 'FontSize', 12);
    legend([h_prob2, h_V2], {'|\psi|^2', 'Re[V(x,t)]'}, 'Location', 'best');
    grid on;
    xlim([x(1), x(end)]);
    
    subplot(2,2,2);
    h_real2 = plot(x, real(psi_history(:,1)), 'g-', 'LineWidth', 1.5);
    hold on;
    h_imag2 = plot(x, imag(psi_history(:,1)), 'm-', 'LineWidth', 1.5);
    h_abs2 = plot(x, abs(psi_history(:,1)), 'k--', 'LineWidth', 1);
    xlabel('Position x', 'FontSize', 12);
    ylabel('Wavefunction \psi', 'FontSize', 12);
    h_title2_2 = title(sprintf('t = %.2f s', t(1)), 'FontSize', 12);
    legend([h_real2, h_imag2, h_abs2], {'Re(\psi)', 'Im(\psi)', '|\psi|'}, 'Location', 'best');
    grid on;
    xlim([x(1), x(end)]);
    ylim(psi_ylim);
    
    subplot(2,2,3);
    h_phi_k2 = plot(k_plot, phi_k_history(:,1), 'b-', 'LineWidth', 2);
    xlabel('Momentum k', 'FontSize', 12);
    ylabel('|\phi(k)|^2', 'FontSize', 12);
    h_title3_2 = title(sprintf('t = %.2f s', t(1)), 'FontSize', 12);
    grid on;
    xlim(k_range);
    ylim(phi_k_ylim);
    
    subplot(2,2,4);
    h_norm2 = plot(t(1), norm_history(1), 'b-', 'LineWidth', 2);
    hold on;
    yline(norm_history(1), 'r--', 'LineWidth', 1);
    xlabel('Time t (s)', 'FontSize', 12);
    ylabel('Total probability ∫|ψ|^2dx', 'FontSize', 12);
    h_title4_2 = title('Probability Evolution', 'FontSize', 12);
    legend(h_norm2, {'Total probability'}, 'Location', 'best');
    grid on;
    xlim([t(1), t(end)]);
    ylim(norm_ylim);
    
    % Main title - auto-updates based on potential function
    sgtitle(sprintf('Time-Dependent Schrödinger Equation Evolution | V(x,t) = %s', V_str), ...
            'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'tex');
    
    % Force draw to confirm figure size
    drawnow;
    
    video_tic = tic;
    for n = 1:params.Nt
        set(h_prob2, 'YData', prob_history(:,n));
        set(h_V2, 'YData', real(V_history(:,n)));
        set(h_real2, 'YData', real(psi_history(:,n)));
        set(h_imag2, 'YData', imag(psi_history(:,n)));
        set(h_abs2, 'YData', abs(psi_history(:,n)));
        set(h_phi_k2, 'YData', phi_k_history(:,n));
        
        set(h_norm2, 'XData', t(1:n), 'YData', norm_history(1:n));
        
        set(h_title1_2, 'String', sprintf('t = %.2f s', t(n)));
        set(h_title2_2, 'String', sprintf('t = %.2f s', t(n)));
        set(h_title3_2, 'String', sprintf('t = %.2f s', t(n)));
        
        drawnow;
        
        % Capture frame and write to video
        frame = getframe(fig2);
        writeVideo(v, frame);
        
        if mod(n, floor(params.Nt/10)) == 0
            fprintf('Video progress: %.0f%%\n', 100*n/params.Nt);
        end
    end
    
    close(v);
    close(fig2);
    video_time = toc(video_tic);
    fprintf('\nVideo saved successfully!\n');
    fprintf('Filename: %s.mp4\n', video_filename);
    fprintf('Time elapsed: %.2fs\n', video_time);
else
    fprintf('Skipping video save.\n');
end

%% ==================== Result Output ====================
fprintf('\n==================== Final Results ====================\n');
fprintf('Potential function: V(x,t) = %s\n', V_str);
fprintf('Initial wavefunction: %s\n', psi0_str);
fprintf('Evolution time: %.1f seconds\n', params.t_range(2));
fprintf('Total frames: %d\n', params.Nt);
fprintf('\nNormalization check:\n');
fprintf('  Initial: ∫|ψ|^2dx = %.8f\n', norm_history(1));
fprintf('  Final: ∫|ψ|^2dx = %.8f\n', norm_history(end));
fprintf('  Change: %.2e\n', norm_history(end) - norm_history(1));

% Check for imaginary potential
if max(abs(imag(V_history(:)))) > 1e-10
    fprintf('  Note: Probability is not strictly conserved due to complex potential.\n');
else
    if abs(norm_history(end) - norm_history(1)) < 1e-6
        fprintf('  ✓ Good probability conservation\n');
    else
        fprintf('  ⚠ Probability deviation detected\n');
    end
end

fprintf('\nPerformance statistics:\n');
fprintf('  Calculation time: %.2fs\n', calc_time);
fprintf('  Playback time: %.2fs\n', play_time);
fprintf('  Actual playback FPS: %.1f FPS\n', params.Nt/play_time);
fprintf('================================================\n');

fprintf('\nProgram execution completed!\n');
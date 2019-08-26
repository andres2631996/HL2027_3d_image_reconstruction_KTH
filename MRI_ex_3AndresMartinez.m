clear
close all
clc
addpath seemri
%% [TODO] define the variables gammabar and gamma
gammabar = 42.58*10^6;
gamma = 2*pi*gammabar;

%% 3. Brain imaging

% Design the RF pulse
B0 = 1.5;
tp = 1e-3;
alpha = pi/2;
% [TODO] Fill in the pulse amplitude and frequence
B1 = alpha/(gamma*tp);
f_rf = gammabar*B0;
rf = RectPulse(B1, f_rf, 0, tp);

% Time parameters
TE = 10e-3;
TR = 2;
tau = 4e-3;

% [TODO] Fill in the pixel size
pixel_size = 2;
% [TODO] Fill in the field of view
FOV = 220;

% Generate the sampling grid
% [TODO] Fill in the kmax
kmax = 1/(2*pixel_size);
% [TODO] Fill in the sampling distance in k-space
dk = 1/FOV;
dk = kmax/ceil(kmax/dk);
ks = -kmax:dk:kmax-dk;

% Phase Encoding Gradient
% [TODO] Fill in the amplitude of the phase encoding gradient Gpexs
Gpexs = ks./(gammabar*tau);
gx = Gradient([tp tp+tau], {Gpexs 0});

% Frequency Encoding Gradient
% [TODO] Fill in the amplitude of the frequency encoding gradient
Gfey1 = -kmax/(gammabar*tau);
Gfey2 = kmax/(gammabar*tau);
gy = Gradient([tp tp+tau TE-tau TE+tau], [Gfey1 0 Gfey2 0]);

% Sampling interval
TE = 10e-3;
TR = 2;
tau = 4e-3;
% [TODO] Fill in the sampling time
dt = 2*tau*pixel_size/FOV;
adc = ADC(TE-tau, TE+tau, dt);

% Visualize the result
[S, ts] = brain_2mm_pixel(B0, rf, gx, gy, adc, TR, length(Gpexs));
figure
mrireconstruct(S, kmax, 'Plot', true);
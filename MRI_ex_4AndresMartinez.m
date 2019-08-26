clear
close all
clc

%% [TODO] define the variables gammabar and gamma
gammabar = 42.58*10^6;
gamma = 2*pi*gammabar;

%% [TODO] Get the missing data from the previous exercise:

B0 = 1.5;
tp = 1e-3;
alpha = pi/2;
B1 = alpha/(gamma*tp);
f_rf = gammabar*B0;
rf = RectPulse(B1, f_rf, 0, tp);

pixel_size = 2;
FOV = 220;

kmax = 1/(2*pixel_size);
dk = 1/FOV;
dk = kmax/ceil(kmax/dk);
ks = -kmax:dk:kmax-dk;

%% T1-weighted

tau = 2e-3;
TE = 5e-3;
% [TODO] Test different repetition times
TR = 10;

% [TODO] Get the amplitude of the phase and frequency encoding gradients from the previous exercise
Gpexs = ks/(gammabar*tau);
gx = Gradient([tp tp+tau], {Gpexs 0});
% Frequency Encoding Gradient
Gfey1 = -kmax/(gammabar*tau);
Gfey2 = kmax/(gammabar*tau);
gy = Gradient([tp tp+tau TE-tau TE+tau], [Gfey1 0 Gfey2 0]);
% Sampling interval
dt = (tau*2*pixel_size)/FOV;
adc = ADC(TE-tau, TE+tau, dt);

[S, ts] = brain_2mm_pixel_fuzzy(B0, rf, gx, gy, adc, TR, length(Gpexs));
figure
mrireconstruct(S, kmax, 'Plot', true)
title('T_1 - weighted')

%% T2-weighted

tau = 2e-3;
TR = 5;
% [TODO] Test different echo times
TE = 40e-3;

% Get the amplitude of the phase and frequency encoding gradients from the
% previous exercise
Gpexs = ks/(gammabar*tau);
gx = Gradient([tp tp+tau], {Gpexs 0});
% Frequency Encoding Gradient
Gfey1 = -kmax/(gammabar*tau);
Gfey2 = kmax/(gammabar*tau);
gy = Gradient([tp tp+tau TE-tau TE+tau], [Gfey1 0 Gfey2 0]);
% Sampling interval
dt = (tau*2*pixel_size)/FOV;
adc = ADC(TE-tau, TE+tau, dt);

[S, ts] = brain_2mm_pixel_fuzzy(B0, rf, gx, gy, adc, TR, length(Gpexs));
figure
mrireconstruct(S, kmax, 'Plot', true)
title('T_2 - weighted')
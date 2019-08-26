clear
close all
clc
addpath seemri
%% [TODO] define the variables gammabar and gamma
gammabar = 42.58*10^6;
gamma = 2*pi*gammabar;

%% 1. Designing a slice selection sequence

B0 = 3;
iv1 = ImagingVolume(-2:0.25:2, 1:0.25:7, 0.8, 0.1, 1, 'PlotScale', 2);
B1 = 2e-6;

% [TODO] Fill in pulse duration tp
tp = 3*10^(-3);
% [TODO] Fill in the bandwidth of the pulse Delta_f
Delta_f = 12/tp;

% [TODO] Fill in the slice thickness Delta_s
Delta_s = 2*10^(-3);

% [TODO] Fill in the amplitude of the gradient Gss
Gss = Delta_f/(gammabar*Delta_s);




% implementation of the gradient:
g = Gradient([0 tp 1.5*tp], [Gss -Gss 0]);

% [TODO] Fill in the center of the slice y0
y0 = 4*10^(-3);

% [TODO] Fill in the frequency of the pulse f_rf
f_rf = gammabar*B0;
% implementation of the RF pulse:
rf = SincPulse(B1, f_rf, 0, tp);

% Apply the pulse sequence
iv1.toEquilibrium();
[S, ts] = seemri(iv1, B0, rf, [], g, ADC(1.5*tp, tp/100));

% Apply the new pulse with gradients on both x- and y- directions.
iv1.toEquilibrium();
% [TODO] fill in the missing parameters
[S, ts] = seemri(iv1,B0,rf,g,g,ADC(1.5*tp, tp/100));
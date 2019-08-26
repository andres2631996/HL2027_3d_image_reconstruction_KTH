clear
close all
clc
addpath seemri
%% [TODO] define the variables gammabar and gamma
gammabar = 42.58*10^6;
gamma = 2*pi*gammabar;

%% 2. K-space sampling

% Definition of the image
u = @(x,y) ((x.^2)/4 + y.^2 )<=1; %ellipse with axis 2 and 1

% Visualize the image
x = -5:0.1:5;
y = -5:0.1:5;
[xs, ys] = meshgrid(x, y);
figure
imagesc(x, y, u(xs, ys));
axis image
colormap gray
title('Image space')

% Continuous FT formulation
Fu = @(kx,ky) 2*besselj(1, 2*pi*(sqrt((2*kx).^2+ky.^2)+1e-9*(kx==0 & ky==0)))...
./(sqrt((2*kx).^2+ky.^2)+1e-9*(kx==0 & ky==0));

% Design of the sampling grid
dk = 0.05;
kmax = 5;
ks = -kmax:dk:kmax-dk;
[kxs, kys] = meshgrid(ks,ks);

% View the k-space
figure
imagesc(ks, ks, Fu(kxs, kys));
axis image
colormap gray
title('k-space')

% Reconstruct and view the image
figure
mrireconstruct(Fu(kxs, kys), kmax, 'Plot', true)
title(sprintf('kmax = %g, dk = %g', kmax, dk))

% Reconstruct and view the image again varying dk and kmax!
for dk=[0.1,0.4]
   kmax = 5;
   ks = -kmax:dk:kmax-dk;
   [kxs, kys] = meshgrid(ks,ks);
   % View the k-space
    figure
    imagesc(ks, ks, Fu(kxs, kys));
    axis image
    colormap gray
    title('k-space')

    % Reconstruct and view the image
    figure
    mrireconstruct(Fu(kxs, kys), kmax, 'Plot', true)
    title(sprintf('kmax = %g, dk = %g', kmax, dk))
end

for kmax=[2,4,10]
   dk=0.05;
   ks = -kmax:dk:kmax-dk;
   [kxs, kys] = meshgrid(ks,ks);
   % View the k-space
    figure
    imagesc(ks, ks, Fu(kxs, kys));
    axis image
    colormap gray
    title('k-space')

    % Reconstruct and view the image
    figure
    mrireconstruct(Fu(kxs, kys), kmax, 'Plot', true)
    title(sprintf('kmax = %g, dk = %g', kmax, dk))
end
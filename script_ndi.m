%--------------------------------------------------------------------------
%% load data 
%--------------------------------------------------------------------------

load mask_use
load mgn_use
load phase_use

load kernel
load scale_factor       % TE * B0 * gyro

load chi_cosmos_5dir
load chi_ndi_5dir


N = size(mask_use);


%--------------------------------------------------------------------------
%% TKD
%--------------------------------------------------------------------------

kthre = 0.15;       % truncation threshold for TKD recon

krnl = kernel(:,:,:,1);
kernel_inv = zeros(N);
kernel_inv( abs(krnl) > kthre ) = 1 ./ krnl(abs(krnl) > kthre);

tic
    chi_tkd = real( ifftn( fftn(phase_use(:,:,:,1)) .* kernel_inv ) ) .* mask_use; 
toc

rmse_tkd = rmse(chi_tkd, chi_cosmos);
rmse_tkd_nl = rmse(chi_tkd, chi_ndi_5dir .* mask_use / scale_factor);

disp(['TKD rmse wrt 5-dir Cosmos: ', num2str(rmse_tkd), '%   rmse wrt 5-dir NDI: ', num2str(rmse_tkd_nl), '%'])

imagesc3d2(chi_tkd, N/2, 1, [90,90,-90], [-0.1,.1], [], 'tkd')


    
%--------------------------------------------------------------------------
%% L2 recon with gradient smoothness constraint
%--------------------------------------------------------------------------

params = [];
params.N = N;                           % Number of voxels
params.D = krnl;                           % Dipole kernel
params.phase_unwrap = phase_use(:,:,:,1);            % V-SHARP filtered phase


params.alpha = 4e-2;                    % Regularization param for quadratic smoothing

tic
    out_ms_l2 = MS_L2_QSM(params);
toc

chi_l2 = out_ms_l2.x .* mask_use;

 
rmse_l2 = rmse(chi_l2, chi_cosmos);
rmse_l2_nl = rmse(chi_l2, chi_ndi_5dir .* mask_use / scale_factor);


disp(['L2 rmse wrt 5-dir Cosmos: ', num2str(rmse_l2), '%   rmse wrt 5-dir NDI: ', num2str(rmse_l2_nl), '%'])


imagesc3d2(chi_l2, N/2, 2, [90,90,-90], [-0.1,.1], [], 'l2')



%--------------------------------------------------------------------------
%% FANSI
%--------------------------------------------------------------------------

params = [];

params.maxOuterIter = 100;
params.tol_update = 1;

params.N = N;
params.K = krnl;
params.input = phase_use(:,:,:,1);

params.mu2 = 1.0;                   % data consistency
params.mu1 = 1e-2;                  % gradient consistency

mgn = mgn_use(:,:,:,1);
params.weight = mgn .* mask_use / max(mgn(:) .* mask_use(:));

    
params.alpha1 = 4e-5;               % gradient L1 penalty
    
tic
    outnl = nlTV(params); 
toc
    
Chi_fansi = real(outnl.x) .* mask_use;

 
rmse_fansi = rmse(Chi_fansi, chi_cosmos);
rmse_fansi_nl = rmse(Chi_fansi, chi_ndi_5dir .* mask_use / scale_factor);

disp(['FANSI rmse wrt 5-dir Cosmos: ', num2str(rmse_fansi), '%   rmse wrt 5-dir NDI: ', num2str(rmse_fansi_nl), '%'])


imagesc3d2(Chi_fansi, N/2, 3, [90,90,-90], [-0.1,.1], [], 'Fansi')
 
 

%--------------------------------------------------------------------------
%% NDI
%--------------------------------------------------------------------------

step_size = 1;      % gradient descent step size
num_iter = 200;    

nd = 1;             % number of head directions to use in the NDI recon, can be set between 1 and 5

phs_use = phase_use(:,:,:,1:nd) .* scale_factor;

M2 = repmat(mean(mgn_use(:,:,:,1:nd),4).^2, [1,1,1,nd]);        % magnitude weighting


if nd == 1
    tol = 0.5;      % 1-direction may require more iterations: decrease convergence tolerance
else
    tol = 1;
end

Chi = zeross(N);
grad_prev = 0;

tic
for t = 1:num_iter
    temp = M2 .* sin(ifft(ifft(ifft(kernel(:,:,:,1:nd) .* repmat(fftn(Chi),[1,1,1,nd]), [], 1), [], 2), [], 3) - phs_use);

    grad_f = 2 * sum(ifft(ifft(ifft(kernel(:,:,:,1:nd) .* fft(fft(fft(temp, [], 1), [], 2), [], 3), [], 1), [], 2), [], 3), 4);

    Chi = Chi - step_size * real(grad_f);

    update_grad = rmse(grad_prev, grad_f);

    disp(['iter: ', num2str(t), '   grad update:', num2str(update_grad)])

    if update_grad < tol
        break
    end

    grad_prev = grad_f;
end
toc

rmse_ndi = rmse(Chi .* mask_use / scale_factor, chi_cosmos);
rmse_ndi_nl = rmse(Chi .* mask_use, chi_ndi_5dir .* mask_use);

disp(['NDI rmse wrt 5-dir Cosmos: ', num2str(rmse_ndi), '%   rmse wrt 5-dir NDI: ', num2str(rmse_ndi_nl), '%'])

 
imagesc3d2(Chi .* mask_use / scale_factor, N/2, 4, [90,90,-90], [-0.1,0.1], [], 'NDI')


 
 

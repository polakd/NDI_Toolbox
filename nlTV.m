function out = nlTV(params)

mu = params.mu1; % gradient consistency
lambda = params.alpha1;  % gradient L1 penalty
    mu2 = params.mu2;
N = params.N;

    if isfield(params,'maxOuterIter')
        num_iter = params.maxOuterIter;
    else
        num_iter = 100;
    end
tol_update = params.tol_update;

    if isfield(params,'weight')
        weight = params.weight;
    else
        weight = ones(N);
    end
    
    if ~isfield(params,'delta_tol')
        delta_tol = 1e-6;
    else
        delta_tol = params.delta_tol;
    end
% 
    
    phase = params.input;
    W = weight.*weight;
    
z_dx = zeros(N, 'single');
z_dy = zeros(N, 'single');
z_dz = zeros(N, 'single');

s_dx = zeros(N, 'single');
s_dy = zeros(N, 'single');
s_dz = zeros(N, 'single');

x = zeros(N, 'single');
    s2 = zeros(N,'single'); z2 = W.*phase./(W+mu2); %zeros(N,'single');

    kernel = params.K;


[k1, k2, k3] = ndgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);

E1 = 1 - exp(2i .* pi .* k1 / N(1));
E2 = 1 - exp(2i .* pi .* k2 / N(2));
E3 = 1 - exp(2i .* pi .* k3 / N(3));

E1t = conj(E1);
E2t = conj(E2);
E3t = conj(E3);

EE2 = E1t .* E1 + E2t .* E2 + E3t .* E3;
K2 = abs(kernel).^2;

tic
for t = 1:num_iter
    % update x : susceptibility estimate
    tx = E1t .* fftn(z_dx - s_dx);
    ty = E2t .* fftn(z_dy - s_dy);
    tz = E3t .* fftn(z_dz - s_dz);
    
    x_prev = x;
    Dt_kspace = conj(kernel) .* fftn(z2-s2);
    x = real(ifftn( (mu * (tx + ty + tz) + Dt_kspace) ./ (eps + mu2*K2 + mu * EE2) ));

    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    disp(['Iter: ', num2str(t), '   Update: ', num2str(x_update)])
    
    if x_update < tol_update
        break
    end
    
    if t < num_iter
        % update z : gradient varible
        Fx = fftn(x);
        x_dx = real(ifftn(E1 .* Fx));
        x_dy = real(ifftn(E2 .* Fx));
        x_dz = real(ifftn(E3 .* Fx));
        
        ll = lambda/mu;
        z_dx = max(abs(x_dx + s_dx) - ll, 0) .* sign(x_dx + s_dx);
        z_dy = max(abs(x_dy + s_dy) - ll, 0) .* sign(x_dy + s_dy);
        z_dz = max(abs(x_dz + s_dz) - ll, 0) .* sign(x_dz + s_dz);
    
        % update s : Lagrange multiplier
        s_dx = s_dx + x_dx - z_dx;
        s_dy = s_dy + x_dy - z_dy;            
        s_dz = s_dz + x_dz - z_dz;  
        
        
        rhs_z2 = mu2*real(ifftn(kernel.*Fx)+s2  );
        z2 =  rhs_z2 ./ mu2 ;

        % Newton-Raphson method
        delta = inf;
        inn = 0;
        while (delta > delta_tol && inn < 50)
            inn = inn + 1;
            norm_old = norm(z2(:));
            
            update = ( W .* sin(z2 - phase) + mu2*z2 - rhs_z2 ) ./ ( W .* cos(z2 - phase) + mu2 );            
        
            z2 = z2 - update;     
            delta = norm(update(:)) / norm_old;
        end        
        disp(delta)
        
        s2 = s2 + real(ifftn(kernel.*Fx)) - z2;
    end
end
toc

out.x = x;
out.iter = t;



end

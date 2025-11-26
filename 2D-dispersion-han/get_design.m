function design = get_design(design_name,N_pix)
    [struct_rng_seed,num_status] = str2num(num2str(design_name));
    if num_status % if design_name is a number, then I assume I want a random design given by that seed
        rng(struct_rng_seed,'twister')        
        design(:,:,1) = round(rand(N_pix));
        design(:,:,2) = design(:,:,1);
        design(:,:,3) = .6*ones(N_pix);
    else
        switch design_name
            case 'dispersive-tetragonal'
                % Dispersive cell - Tetragonal
                design(:,:,1) = zeros(N_pix); % the first pane is E
                idxs = round(N_pix/4 + 1):round(3*N_pix/4);
                design(idxs,idxs,1) = 1;
                design(:,:,2) = design(:,:,1); % the second pane is rho
                design(:,:,3) = .6*ones(N_pix); % the third pane is poisson's ratio
            case 'dispersive-tetragonal-negative'
                % Dispersive cell - Tetragonal
                design(:,:,1) = zeros(N_pix); % the first pane is E
                idxs = round(N_pix/4 + 1):round(3*N_pix/4);
                design(idxs,idxs,1) = 1;
                design(:,:,2) = design(:,:,1); % the second pane is rho
                design(:,:,3) = .6*ones(N_pix); % the third pane is poisson's ratio
                design(:,:,1) = ~design(:,:,1); % negative!
                design(:,:,2) = ~design(:,:,2); % negative!
            case 'dispersive-orthotropic'
                % Dispersive cell - Orthotropic
                design(:,:,1) = zeros(N_pix); % the first pane is E
                idxs = (N_pix/4 + 1):(3*N_pix/4);
                design(:,idxs,1) = 1;
                design(:,:,2) = design(:,:,1); % the second pane is rho
                design(:,:,3) = .6*ones(N_pix); % the third pane is poisson's ratio
            case 'homogeneous'
                % Homogeneous cell
                design(:,:,1) = ones(N_pix); % the first pane is E
                design(:,:,2) = design(:,:,1); % the second pane is rho
                design(:,:,3) = .6*ones(N_pix); % the third pane is poisson's ratio
            case 'quasi-1D'
                % Quasi-1D cell
                design(:,:,1) = ones(N_pix);
                design(:,1:2:end,1) = 0;
                design(:,:,2) = design(:,:,1);
                design(:,:,3) = .6*ones(N_pix);
            case 'rotationally-symmetric'
                design(:,:,1) = zeros(N_pix);
                idxs = (N_pix/4 + 1):(2*N_pix/4);
                design(idxs,idxs,1) = 1;
                idxs = (2*N_pix/4 + 1):(3*N_pix/4);
                design(idxs,idxs,1) = 1;
                design(:,:,2) = design(:,:,1); % the second pane is rho
                design(:,:,3) = .6*ones(N_pix); % the third pane is poisson's ratio
            case 'dirac?'
                design(:,:,1) = zeros(5);
                design([2 3 4 6 8 10 11 12 13 14 15 16 18 20 22 23 24]) = 1;
                design(:,:,2) = design(:,:,1);
                design(:,:,3) = .6*ones(N_pix);
            case 'correlated'
                temp = load("C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\GPR-dispersion-paper\gpr-paper\fig\code_for_figures\elastic_modulus_2D.mat");
                design(:,:,1) = temp.prop;
                temp = load("C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\GPR-dispersion-paper\gpr-paper\fig\code_for_figures\density_2D.mat");
                design(:,:,2) = temp.prop;
                design(:,:,3) = .6*ones(size(design(:,:,1)));
            otherwise
                error(['design not recognized: ' design_name])
        end
    end
end
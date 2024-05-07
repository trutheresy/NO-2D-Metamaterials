clear; close all;

isSaveOutput = true;
isSaveEigenvectors = false;
imag_tol = 1e-3;

const.a = 1; % [m]
const.N_ele = 4;
const.N_pix = 10;
const.N_k = 51;
const.N_eig = 20;
const.isUseGPU = false;
const.isUseImprovement = true;
const.isUseParallel = true;

const.E_min = 2e9;
const.E_max = 200e9;
const.rho_min = 1e3;
const.rho_max = 8e3;
const.poisson_min = 0;
const.poisson_max = .5;
const.t = 1;
const.sigma_eig = 1;

symmetry_type = 'none'; IBZ_shape = 'rectangle'; num_tesselations = 1;
const.wavevectors = create_IBZ_boundary_wavevectors(const.N_k,const.a);

%% Set up save locations
if ~isSaveOutput
    disp('WARNING: isSaveOutput is set to false. Output will not be saved.')
end
script_start_time = replace(char(datetime),':','-');
if isSaveOutput
    output_folder = ['OUTPUT/output ' script_start_time];
    mkdir(output_folder);
    copyfile([mfilename('fullpath') '.m'],[output_folder '/' mfilename '.m']);
end

plot_wavevectors(const.wavevectors)

N_struct = 2^15;

%% Generate dataset
pfwb = parfor_wait(N_struct,'Waitbar', true);
struct_idx = 0;
for i1 = [0 1]
    for i2 = [0 1]
        for i3 = [0 1]
            for i4 = [0 1]
                for i5 = [0 1]
                    for i6 = [0 1]
                        for i7 = [0 1]
                            for i8 = [0 1]
                                for i9 = [0 1]
                                    for i10 = [0 1]
                                        for i11 = [0 1]
                                            for i12 = [0 1]
                                                for i13 = [0 1]
                                                    for i14 = [0 1]
                                                        for i15 = [0 1]
                                                            struct_idx = struct_idx + 1;
                                                            vec = [i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 i12 i13 i14 i15];
                                                            subdesign = zeros(5);
                                                            subdesign(triu(true(5))) = vec;
                                                            subdesign_diag = diag(diag(subdesign));
                                                            subdesign = subdesign + subdesign' - subdesign_diag;
                                                            const.design(:,:,1) = [subdesign, fliplr(subdesign); flipud(subdesign), flipud(fliplr(subdesign))]; % Young's Modulus
                                                            const.design(:,:,2) = const.design(:,:,1); % Density
                                                            const.design(:,:,3) = .3; % Poisson's Ratio
                                                            designs(:,:,:,struct_idx) = const.design;
                                                            
                                                            % Solve the dispersion problem
                                                            [wv,fr,ev] = dispersion(const,const.wavevectors);
                                                            WAVEVECTOR_DATA(:,:,struct_idx) = wv;
                                                            EIGENVALUE_DATA(:,:,struct_idx) = real(fr);
                                                            if isSaveEigenvectors
                                                                EIGENVECTOR_DATA(:,:,:,struct_idx) = ev;
                                                            end
                                                            
                                                            if max(max(abs(imag(fr))))>imag_tol
                                                                warning(['Large imaginary component in frequency for structure ' num2str(struct_idx)])
                                                            end
                                                            
                                                            % Save the material properties
                                                            ELASTIC_MODULUS_DATA(:,:,struct_idx) = const.E_min + (const.E_max - const.E_min)*const.design(:,:,1);
                                                            DENSITY_DATA(:,:,struct_idx) = const.rho_min + (const.rho_max - const.rho_min)*const.design(:,:,2);
                                                            POISSON_DATA(:,:,struct_idx) = const.poisson_min + (const.poisson_max - const.poisson_min)*const.design(:,:,3);
                                                            pfwb.Send; %#ok<PFBNS>
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

pfwb.Destroy;

%% Save the results
if isSaveOutput
    CONSTITUTIVE_DATA = containers.Map({'modulus','density','poisson'},...
        {ELASTIC_MODULUS_DATA, DENSITY_DATA, POISSON_DATA});
    output_file_path = [output_folder '/DATA N_struct' num2str(N_struct) ' N_k' num2str(const.N_k) ' ' script_start_time '.mat'];
    if isSaveEigenvectors
        save(output_file_path,'WAVEVECTOR_DATA','EIGENVALUE_DATA','EIGENVECTOR_DATA','CONSTITUTIVE_DATA','-v7.3');
    else
        save(output_file_path,'WAVEVECTOR_DATA','EIGENVALUE_DATA','CONSTITUTIVE_DATA','-v7.3');
    end
end

%% Plot a subset of the data
% struct_idx_to_plot = 2;
% 
% fig = figure2();
% ax = axes(fig);
% plot_design(cat(3,squeeze(ELASTIC_MODULUS_DATA(:,:,struct_idx_to_plot)), squeeze(DENSITY_DATA(:,:,struct_idx_to_plot)), squeeze(POISSON_DATA(:,:,struct_idx_to_plot))))
% 
% fig = figure2();
% ax = axes(fig);
% hold on
% for eig_idx_to_plot = 4%1:const.N_eig
%     wv_plot = squeeze(WAVEVECTOR_DATA(:,:,struct_idx_to_plot));
%     fr_plot = squeeze(EIGENVALUE_DATA(:,eig_idx_to_plot,struct_idx_to_plot));
%     plot_dispersion_surface(wv_plot,fr_plot,IBZ_shape,ax);
% end
% view(3)

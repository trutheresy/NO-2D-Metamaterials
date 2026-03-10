repo_root = 'D:/Research/NO-2D-Metamaterials';
addpath(fullfile(repo_root, '2D-dispersion-mat'));
inp = fullfile(repo_root, 'OUTPUT', 'recovered_failure_geometries_from_batched_20260303_172713', 'cause_input_failures.mat');
outp = fullfile(repo_root, 'OUTPUT', 'recovered_failure_geometries_from_batched_20260303_172713', 'cause1_matlab_fr_failures.mat');
S = load(inp);
designs = S.designs;
const_base = S.const_base;
N = size(designs,4);
const_base.N_wv = [25 13];
const_base.N_eig = 6;
const_base.sigma_eig = 1e-2;
const_base.symmetry_type = 'p4mm';
const_base.isUseGPU = false;
const_base.isUseImprovement = true;
const_base.isUseSecondImprovement = false;
const_base.isUseParallel = false;
const_base.isSaveEigenvectors = false;
const_base.isSaveMesh = false;
const_base.isSaveKandM = false;
const_base.wavevectors = get_IBZ_wavevectors(const_base.N_wv,const_base.a,const_base.symmetry_type,1);
const_base.wavevectors = const_base.wavevectors(1:55,:);
n_wv = size(const_base.wavevectors,1);
fr = nan(n_wv, const_base.N_eig, N);
status = false(N,1);
messages = cell(N,1);
for i = 1:N
    const = const_base;
    const.N_ele = double(const.N_ele);
    const.N_pix = double(const.N_pix);
    const.a = double(const.a);
    const.E_min = double(const.E_min);
    const.E_max = double(const.E_max);
    const.rho_min = double(const.rho_min);
    const.rho_max = double(const.rho_max);
    const.poisson_min = double(const.poisson_min);
    const.poisson_max = double(const.poisson_max);
    const.t = double(const.t);
    const.design = designs(:,:,:,i);
    try
        [~,fr_i,~,~,~,~,~] = dispersion_with_matrix_save_opt(const,const.wavevectors);
        fr(:,:,i) = real(fr_i);
        status(i) = true;
        messages{i} = 'ok';
    catch ME
        status(i) = false;
        messages{i} = ME.message;
    end
end
save(outp,'fr','status','messages','-v7');
fprintf('Saved MATLAB frequency baseline (failures) to %s\n', outp);

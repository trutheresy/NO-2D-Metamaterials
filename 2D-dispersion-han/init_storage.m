function [designs,WAVEVECTOR_DATA,EIGENVALUE_DATA,N_dof,DESIGN_NUMBERS,EIGENVECTOR_DATA,ELASTIC_MODULUS_DATA,DENSITY_DATA,POISSON_DATA,K_DATA,M_DATA,T_DATA] = init_storage(const,N_struct_batch)
    % Initialize storage arrays
    designs = zeros(const.N_pix,const.N_pix,3,N_struct_batch);
    % WAVEVECTOR_DATA = zeros(prod(const.N_wv),2,N_struct);
    WAVEVECTOR_DATA = zeros(size(const.wavevectors,1),2,N_struct_batch);
    % EIGENVALUE_DATA = zeros(prod(const.N_wv),const.N_eig,N_struct);
    EIGENVALUE_DATA = zeros(size(const.wavevectors,1),const.N_eig,N_struct_batch);
    N_dof = 2*(const.N_pix*const.N_ele)^2;
    % EIGENVECTOR_DATA = zeros(N_dof,prod(const.N_wv),const.N_eig,N_struct,const.eigenvector_dtype); % NEW: Save eigenvectors as either 'single' or 'double' datatype
    DESIGN_NUMBERS = 1:N_struct_batch;
    EIGENVECTOR_DATA = zeros(N_dof,size(const.wavevectors,1),const.N_eig,N_struct_batch,const.eigenvector_dtype); % NEW: Save eigenvectors as either 'single' or 'double' datatype
    ELASTIC_MODULUS_DATA = zeros(const.N_pix,const.N_pix,N_struct_batch);
    DENSITY_DATA = zeros(const.N_pix,const.N_pix,N_struct_batch);
    POISSON_DATA = zeros(const.N_pix,const.N_pix,N_struct_batch);
    % NEW: Initialize storage for stiffness, mass, and transformation matrices
    K_DATA = cell(N_struct_batch,1);
    M_DATA = cell(N_struct_batch,1);
    T_DATA = []; % Optimize by using same wavevector grid for all designs
end
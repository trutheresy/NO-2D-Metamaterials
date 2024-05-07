function [wv,fr,ev,cg,dfrddesign,dcgddesign] = dispersion2(const,wavevectors)
    
    orig_wavevectors = wavevectors;
    
    [wavevectors,~,unique_wavevector_idxs] = unique(wavevectors,'stable','rows'); % Don't repeat computations where unnecessary
    
    fr = zeros(size(wavevectors,1),const.N_eig);
    if const.isSaveEigenvectors
        ev = zeros(((const.N_ele*const.N_pix(1))^2)*2,size(wavevectors,1),const.N_eig);
    else
        ev = [];
    end
    if const.isComputeGroupVelocity
        cg = zeros(size(wavevectors,1),2,const.N_eig);
    end
    
    design_size = size(const.design);
    design_size(3) = design_size(3) - 1; % Poisson's ratio is not a design variable here
    if const.isComputeFrequencyDesignSensitivity || const.isComputeGroupVelocityDesignSensitivity
        dfrddesign = zeros([size(wavevectors,1) const.N_eig design_size]);
    end
    
    if const.isComputeGroupVelocityDesignSensitivity
        dcgddesign = zeros([size(wavevectors) const.N_eig design_size]);
    end
    
    if const.isUseImprovement
        [K,M] = get_system_matrices_VEC(const);
    else
        if const.isComputeFrequencyDesignSensitivity || const.isComputeGroupVelocityDesignSensitivity
            [K,M,dKddesign,dMddesign] = get_system_matrices(const);
        else
            [K,M] = get_system_matrices(const);
        end
    end
    if const.isUseParallel
        parforArg = Inf;
    else
        parforArg = 0;
    end
    for k_idx = 1:size(wavevectors,1) % USE THIS TO DEBUG
%     parfor (k_idx = 1:size(wavevectors,1), parforArg)
        wavevector = wavevectors(k_idx,:);
        if const.isComputeGroupVelocity || const.isComputeGroupVelocityDesignSensitivity
            [T,dTdwavevector] = get_transformation_matrix(wavevector,const);
        else
            T = get_transformation_matrix(wavevector,const);
        end
        
        Kr = T'*K*T;
        Mr = T'*M*T;
        
        if const.isComputeGroupVelocity || const.isComputeGroupVelocityDesignSensitivity
%             dKrdwavevector = ndSparse.build([size(Kr) 2]);
%             dMrdwavevector = ndSparse.build([size(Mr) 2]);
            dKrdwavevector = cell(2,1);
            dMrdwavevector = cell(2,1);
            for wv_comp_idx = 1:2
%                 dKrdwavevector(:,:,wv_comp_idx) = dTdwavevector(:,:,wv_comp_idx)'*K*T + T'*K*dTdwavevector(:,:,wv_comp_idx);
%                 dMrdwavevector(:,:,wv_comp_idx) = dTdwavevector(:,:,wv_comp_idx)'*M*T + T'*M*dTdwavevector(:,:,wv_comp_idx);
%                 dKrdwavevector(:,:,wv_comp_idx) = dTdwavevector{wv_comp_idx}'*K*T + T'*K*dTdwavevector{wv_comp_idx};
%                 dMrdwavevector(:,:,wv_comp_idx) = dTdwavevector{wv_comp_idx}'*M*T + T'*M*dTdwavevector{wv_comp_idx};
                dKrdwavevector{wv_comp_idx} = dTdwavevector{wv_comp_idx}'*K*T + T'*K*dTdwavevector{wv_comp_idx};
                dMrdwavevector{wv_comp_idx} = dTdwavevector{wv_comp_idx}'*M*T + T'*M*dTdwavevector{wv_comp_idx};
            end
        end
        
        % DONT USE THE GPU WITH EIGS
        [eig_vecs,eig_vals] = eigs(Kr,Mr,const.N_eig,const.sigma_eig);
        [eig_vals,idxs] = sort(diag(eig_vals));
        eig_vecs = eig_vecs(:,idxs);
        
        %         ev(:,k_idx,:) = (eig_vecs./vecnorm(eig_vecs,2,1)).*exp(-1i*angle(eig_vecs(1,:))); % normalize by p-norm
        %         eig_vecs = (eig_vecs./vecnorm(eig_vecs,2,1));
        
        eig_vecs = (eig_vecs./sqrt(diag(eig_vecs'*Mr*eig_vecs)')); % Normalize by mass matrix
        eig_vecs = eig_vecs.*exp(-1i*angle(eig_vecs(1,:))); % Align complex angle
        if const.isSaveEigenvectors
            ev(:,k_idx,:) = eig_vecs;
        end
        
        if any(real(eig_vals)<-1e-1)
            warning('large negative eigenvalue')
        end
        fr(k_idx,:) = sqrt(abs(real(eig_vals))); % take square root of eigenvalue
        %         fr(k_idx,:) = fr(k_idx,:)/(2*pi); % convert angular frequency to hertz
        
        if const.isComputeGroupVelocity
            for wv_comp_idx = 1:2
                for eig_idx = 1:const.N_eig
                    omega = fr(k_idx,eig_idx);
                    u = eig_vecs(:,eig_idx);
%                     cg(k_idx,wv_comp_idx,eig_idx) = full(1/(2*omega).*(u'*(dKrdwavevector(:,:,wv_comp_idx) - omega^2*dMrdwavevector(:,:,wv_comp_idx))*u))/(eig_vecs(:,eig_idx)'*Mr*eig_vecs(:,eig_idx));
                    cg(k_idx,wv_comp_idx,eig_idx) = full(1/(2*omega).*(u'*(dKrdwavevector{wv_comp_idx} - omega^2*dMrdwavevector{wv_comp_idx})*u))/(eig_vecs(:,eig_idx)'*Mr*eig_vecs(:,eig_idx));
                    imag_tol = 1e-6;
                    %                     if abs(imag(cg(k_idx,wv_comp_idx,eig_idx))/real(cg(k_idx,wv_comp_idx,eig_idx))) > imag_%                         warning('cg has large imaginary component')
                    %                            warning('cg has large imaginary component')
                    %                     end
                    cg(k_idx,wv_comp_idx,eig_idx) = real(cg(k_idx,wv_comp_idx,eig_idx));
                end
            end
        else
            cg = [];
        end
        
        if const.isComputeFrequencyDesignSensitivity || const.isComputeGroupVelocityDesignSensitivity
            design_size = size(const.design);
            design_size(3) = design_size(3) - 1; % Poisson's ratio is not a design variable here
%             dKrddesign = ndSparse.build([size(Kr) design_size]);
%             dMrddesign = ndSparse.build([size(Mr) design_size]);
            dKrddesign = cell(design_size(1:2));
            dMrddesign = cell(design_size(1:2));
            for i = 1:design_size(1)
                for j = 1:design_size(2)
                    for k = 1:design_size(3)
                        if k == 1 % The design parameter is an elastic modulus parameter
                            %                             dKrddesign(:,:,i,j,k) = T'*dKddesign(:,:,i,j,k)*T;
%                             dKrddesign(:,:,i,j) = T'*dKddesign{i,j}*T;
                            dKrddesign{i,j} = T'*dKddesign{i,j}*T;
                            for eig_idx = 1:const.N_eig
                                omega = fr(k_idx,eig_idx);
                                u = eig_vecs(:,eig_idx);
                                %                                 dfrddesign(k_idx,eig_idx,i,j,k) = 1/(2*omega) * u'*(dKrddesign(:,:,i,j,k) - omega^2*dMrddesign(:,:,i,j,k))*u;
%                                 dfrddesign(k_idx,eig_idx,i,j,k) = 1/(2*omega) * u'*(dKrddesign(:,:,i,j))*u;
                                dfrddesign(k_idx,eig_idx,i,j,k) = 1/(2*omega) * u'*(dKrddesign{i,j})*u;
                                imag_tol = 1e-6;
                                if abs(imag(dfrddesign(k_idx,eig_idx,i,j,k))/real(dfrddesign(k_idx,eig_idx,i,j,k))) > imag_tol
                                    warning('dfrddesign has large imaginary components')
                                end
%                                 dfrddesign(k_idx,eig_idx,i,j,k) = real(dfrddesign(k_idx,eig_idx,i,j,k));
                            end
                        elseif k == 2 % The design parameter is a density parameter
                            %                             dMrddesign(:,:,i,j,k) = T'*dMddesign(:,:,i,j,k)*T;
%                             dMrddesign(:,:,i,j) = T'*dMddesign{i,j}*T;
                            dMrddesign{i,j} = T'*dMddesign{i,j}*T;
                            for eig_idx = 1:const.N_eig
                                omega = fr(k_idx,eig_idx);
                                u = eig_vecs(:,eig_idx);
                                %                                 dfrddesign(k_idx,eig_idx,i,j,k) = 1/(2*omega) * u'*(dKrddesign(:,:,i,j,k) - omega^2*dMrddesign(:,:,i,j,k))*u;
%                                 dfrddesign(k_idx,eig_idx,i,j,k) = 1/(2*omega) * u'*(-omega^2*dMrddesign(:,:,i,j))*u;
                                dfrddesign(k_idx,eig_idx,i,j,k) = 1/(2*omega) * u'*(-omega^2*dMrddesign{i,j})*u;
                                imag_tol = 1e-6;
                                if abs(imag(dfrddesign(k_idx,eig_idx,i,j,k))/real(dfrddesign(k_idx,eig_idx,i,j,k))) > imag_tol
                                    warning('dfrddesign has large imaginary components')
                                end
%                                 dfrddesign(k_idx,eig_idx,i,j,k) = real(dfrddesign(k_idx,eig_idx,i,j,k));
                            end
                        end
                    end
                end
            end
            dfrddesign(k_idx,:,:,:,:) = real(dfrddesign(k_idx,:,:,:,:));
        end
        
        if const.isComputeGroupVelocityDesignSensitivity
            %             if ~any(wavevector == pi || wavevector == -pi)
%             d2Krddesigndwavevector = ndSparse.build([size(dKrdwavevector) design_size]);
%             d2Mrddesigndwavevector = ndSparse.build([size(dMrdwavevector) design_size]);
            for wv_comp_idx = 1:2
                for eig_idx = 1:const.N_eig
                    for i = 1:design_size(1)
                        for j = 1:design_size(2)
                            for k = 1:design_size(3)
                                omega = fr(k_idx,eig_idx);
                                u = eig_vecs(:,eig_idx);
%                                 A = dKrdwavevector(:,:,wv_comp_idx) - omega^2 * dMrdwavevector(:,:,wv_comp_idx);
                                A = dKrdwavevector{wv_comp_idx} - omega^2 * dMrdwavevector{wv_comp_idx};
                                domegadtheta = dfrddesign(k_idx,eig_idx,i,j,k);
%                                 dTdgamma = dTdwavevector(:,:,wv_comp_idx);
                                dTdgamma = dTdwavevector{wv_comp_idx};
                                if k == 1
%                                     dKrdtheta = dKrddesign(:,:,i,j);
                                    dKrdtheta = dKrddesign{i,j};
%                                     dMrdtheta = zeros(size(dMrddesign(:,:,i,j)));
                                    dMrdtheta = zeros(size(dMrddesign{i,j}));
                                elseif k == 2
%                                     dKrdtheta = zeros(size(dKrddesign(:,:,i,j)));
                                    dKrdtheta = zeros(size(dKrddesign{i,j}));
%                                     dMrdtheta = dMrddesign(:,:,i,j);
                                    dMrdtheta = dMrddesign{i,j};
                                end
                                
                                %                                 d2Krddesigndwavevector(:,:,wv_comp_idx,i,j,k) = dTdwavevector(:,:,wv_comp_idx)'*dKddesign(:,:,i,j,k)*T + T'*dKddesign(:,:,i,j,k)*dTdwavevector(:,:,wv_comp_idx);
                                %                                 d2Mrddesigndwavevector(:,:,wv_comp_idx,i,j,k) = dTdwavevector(:,:,wv_comp_idx)'*dMddesign(:,:,i,j,k)*T + T'*dMddesign(:,:,i,j,k)*dTdwavevector(:,:,wv_comp_idx);
                                %                                 dAddesign = d2Krddesigndwavevector(:,:,wv_comp_idx,i,j,k) - 2*omega*dfrddesign(k_idx,eig_idx,i,j,k) - omega^2*d2Mrddesigndwavevector(:,:,wv_comp_idx,i,j,k);
                                if k == 1 % The design parameter is an elastic modulus parameter
                                    dKdtheta = dKddesign{i,j};
                                    %                                     d2Krddesigndwavevector(:,:,wv_comp_idx,i,j,k) = dTdwavevector(:,:,wv_comp_idx)'*dKddesign(:,:,i,j)*T + T'*dKddesign(:,:,i,j)*dTdwavevector(:,:,wv_comp_idx);
                                    d2Krdthetadgamma = dTdgamma'*dKdtheta*T + T'*dKdtheta*dTdgamma;
                                    %                                     dAddesign(:,:,wv_comp_idx,i,j,k) = d2Krddesigndwavevector(:,:,wv_comp_idx,i,j,k) - 2*omega*dfrddesign(k_idx,eig_idx,i,j,k)*dMrdwavevector(:,:,wv_comp_idx) - omega^2*d2Mrddesigndwavevector(:,:,wv_comp_idx,i,j,k);
%                                     dAdtheta = d2Krdthetadgamma - 2*omega*domegadtheta*dMrdwavevector(:,:,wv_comp_idx);
                                    dAdtheta = d2Krdthetadgamma - 2*omega*domegadtheta*dMrdwavevector{wv_comp_idx};
                                elseif k == 2 % The design parameter is a density parameter
                                    dMdtheta = dMddesign{i,j};
                                    %                                     d2Mrddesigndwavevector(:,:,wv_comp_idx,i,j,k) = dTdwavevector(:,:,wv_comp_idx)'*dMddesign(:,:,i,j)*T + T'*dMddesign(:,:,i,j)*dTdwavevector(:,:,wv_comp_idx);
                                    d2Mrdthetadgamma = dTdgamma'*dMdtheta*T + T'*dMdtheta*dTdgamma;
                                    %                                     dAddesign(:,:,wv_comp_idx,i,j,k) = d2Krddesigndwavevector(:,:,wv_comp_idx,i,j,k) - 2*omega*dfrddesign(k_idx,eig_idx,i,j,k)*dMrdwavevector(:,:,wv_comp_idx) - omega^2*d2Mrddesigndwavevector(:,:,wv_comp_idx,i,j,k);
%                                     dAdtheta = - 2*omega*domegadtheta*dMrdwavevector(:,:,wv_comp_idx) - omega^2*d2Mrdthetadgamma;
                                    dAdtheta = - 2*omega*domegadtheta*dMrdwavevector{wv_comp_idx} - omega^2*d2Mrdthetadgamma;
                                end
                                %                                 duddesign = get_duddesign(Kr,Mr,omega,u,dKrddesign(:,:,i,j,k),dMrddesign(:,:,i,j,k));
                                duddesign = get_duddesign(Kr,Mr,omega,u,dKrdtheta,dMrdtheta);
                                
                                term1 = -1/(2*omega^2) * domegadtheta * u'*A*u;
                                term2 = 1/(2*omega) * duddesign'*A*u;
                                term3 = 1/(2*omega) * u'*dAdtheta*u;
                                term4 = 1/(2*omega) * u'*A*duddesign;
                                
                                dcgddesign(k_idx,wv_comp_idx,eig_idx,i,j,k) = term1 + term2 + term3 + term4;
                                imag_tol = 1e-6;
                                if abs(imag(dcgddesign(k_idx,wv_comp_idx,eig_idx,i,j,k))/real(dcgddesign(k_idx,wv_comp_idx,eig_idx,i,j,k)))>imag_tol
                                    warning('dcgddesign has large imaginary components')
                                end
                                dcgddesign(k_idx,wv_comp_idx,eig_idx,i,j,k) = real(dcgddesign(k_idx,wv_comp_idx,eig_idx,i,j,k));
                            end
                        end
                    end
                end
            end
        end
    end
    if const.isComputeGroupVelocity
        cg = cg(unique_wavevector_idxs,:,:);
    end
    if const.isComputeFrequencyDesignSensitivity
        dfrddesign = dfrddesign(unique_wavevector_idxs,:,:,:,:);
    end
    if const.isComputeGroupVelocityDesignSensitivity
        dcgddesign = dcgddesign(unique_wavevector_idxs,:,:,:,:,:);
    end
    fr = fr(unique_wavevector_idxs,:);
    if const.isSaveEigenvectors
        ev = ev(:,unique_wavevector_idxs,:);
    end
    wv = orig_wavevectors;
end
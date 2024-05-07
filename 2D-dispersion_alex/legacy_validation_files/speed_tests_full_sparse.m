clear; close all;

rng(1)

isLoadMatrices = true;
N_pix = 2:1:10;
N_ele = 4;

N_sample = 10;

% simple study
sizes = [10 20 30 40 50 100 200 300 400 500 1000 5000]; % matrix size
rc = 1e-12*ones(length(sizes),1); % reciprocal condition number of random sparse matrices
frac_nz = .05*ones(length(sizes),1); % fraction of nonzero elements

if ~isLoadMatrices
    counter = 0;
    wb = waitbar(0,'Computing...');
    for s = 1:length(sizes)
        disp(sizes(s))
        for i = 1:N_sample
            disp(i)
            
            A = sprand(sizes(s),sizes(s),frac_nz(s),rc(s));
            b = rand(sizes(s),1);
            
            disp('sparse')
            tic
            x = A\b;
            t_sparse(i,s) = toc;
            
            A_f = full(A);
            
            disp('full')
            tic
            x = A_f\b;
            t_full(i,s) = toc;
            
            counter = counter + 1;
            waitbar(counter/(N_sample*length(sizes)),wb)
        end
    end
    close(wb)
end

% if isLoadMatrices
%     clear sizes
%     counter = 0;
%     wb = waitbar(0,'Computing...');
%     for s = 1:length(N_pix)
%         disp(N_pix(s))
%         load(['A_and_b_N_pix' num2str(N_pix(s))])
%         sizes(1,s) = size(A,1);
%
%         disp('sparse')
%         tic
%         x = A\b;
%         t_sparse(1,s) = toc;
%
%         A_f = full(A);
%
%         disp('full')
%         tic
%         x = A_f\b;
%         t_full(1,s) = toc;
%
%         counter = counter + 1;
%         waitbar(counter/length(N_pix),wb)
%     end
% end

if isLoadMatrices
    clear sizes rc frac_nz
    counter = 0;
    wb = waitbar(0,'Computing...');
    for s = 1:length(N_pix)
        disp(N_pix(s))
        load(['A_and_b_N_pix' num2str(N_pix(s)) '_N_ele' num2str(N_ele)])
        sizes(1,s) = size(A,1);
        
        A_orig = A;
        rc(1,s) = rcond(full(A));
        frac_nz(1,s) = nnz(A)/numel(A);
        
        disp('sparse')
        tic
        x = A\b;
        t_sparse(1,s) = toc;
        
        A_f = full(A);
        
        disp('full')
        tic
        x = A_f\b;
        t_full(1,s) = toc;
        
        for i = 2:N_sample
            A = sprand(A_orig);
            
            rc(i,s) = rcond(full(A));
            frac_nz(i,s) = nnz(A)/numel(A);
            
            disp('sparse')
            tic
            x = A\b;
            t_sparse(i,s) = toc;
            
            
            disp('full')
            tic
            A_f = full(A);
            
            x = A_f\b;
            t_full(i,s) = toc;
        end
        
        counter = counter + 1;
        waitbar(counter/length(N_pix),wb)
    end
end
close(wb)

t_sparse_mean = mean(t_sparse,1);

t_full_mean = mean(t_full,1);

figure
hold on
p1 = plot(sizes,t_sparse_mean,'r*');
p2 = plot(sizes,t_full_mean,'k*');
p1.DisplayName = 'sparse';
p2.DisplayName = 'full';
legend('location','NorthWest')
title('solve time vs. matrix size')
xlabel('matrix size (i.e. axis value = N --> matrix is N x N)')
ylabel('time taken to solve linear system')

figure
hold on
for i = 1:N_sample
    if i == 1 && isLoadMatrices
        p3(i) = plot(sizes,t_sparse(i,:),'rs');
        p4(i) = plot(sizes,t_full(i,:),'ks');
    elseif i == 1
        p3(i) = plot(sizes,t_sparse(i,:),'r*');
        p4(i) = plot(sizes,t_full(i,:),'k*');
    else
%         p3(i) = plot(sizes,t_sparse(i,:),'r*');
%         p4(i) = plot(sizes,t_full(i,:),'k*');
    end
    if i == 1
        p3(i).DisplayName = 'sparse';
        p4(i).DisplayName = 'full';
    end
end
legend([p3(1),p4(1)],'location','NorthWest')
title('solve time vs. matrix size')
xlabel('matrix size (i.e. axis value = N --> matrix is N x N)')
ylabel('time taken to solve linear system')

figure
p5 = plot(N_pix,rc(2:end,:),'k*');
p5(1).DisplayName = 'Random matrices with original sparsity structure';
hold on
p6 = plot(N_pix,rc(1,:),'ks');
p6.DisplayName = 'Original matrix from get\_duddesign';
set(gca,'yscale','log')
legend([p5(1) p6],'location','NorthEast')
title(['Reciprocal condition number vs. N\_pix' newline '(larger rcond --> system farther from singular)'])
xlabel('N\_pix')
ylabel('reciprocal condition number')

figure
p7 = plot(N_pix,frac_nz(1,:),'ks');
p7.DisplayName = 'Original matrix from get\_duddesign';
set(gca,'yscale','log')
legend([p7],'location','NorthEast')
title(['Fraction of nonzero elements vs. N\_pix'])
xlabel('N\_pix')
ylabel('Fraction of nonzero elements')
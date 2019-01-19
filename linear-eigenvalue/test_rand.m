function test_rand
% Strutured quasi-Newton method for solving linear eigenvalue problem
%
% (A+B)X = X Lambda, least p eigenvalues and corresponding eigenvectors
%
% The computational cost BX is much higher than AX. B is assumed to be
% negative definite. In this program, A and B are random matrices, but BX  
% is computed as BX = 1/19(B*X + ... + B*X).

% Reference:
%  J. Hu, B. Jiang, L. Lin, Z. Wen and Y. Yuan
%  Structured Quasi-newton Methods for Optimization with
%  Orthogonality Constraints
%
%   Author: J. Hu, Z. Wen
%  Version 1.0 .... 2018/9


Problist = [1];

for dprob = Problist
    % fix seed
    seed = 2018;
    if exist('RandStream','file')
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
    else
        rand('state',seed); randn('state',seed);
    end
    
    filename = strcat('./results', filesep,'Date_',num2str(date),'prob',num2str(dprob),'-1.txt');
    fid = fopen(filename,'w+');
    
    fprintf(fid,'\n');
    
    switch dprob
        case {1}
            plist = 10;
            Nlist = 5000;
            % Nlist = [5000,6000, 8000, 1e4];
        case {2}
            Nlist = 5000;
            plist = [10, 20, 30, 50];
            % plist = [10];
    end
    
    for p = plist
        for N = Nlist
            seed = 2018;
            if exist('RandStream','file')
                RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
            else
                rand('state',seed); randn('state',seed);
            end
            
            
            A = randn(N);
            A = (A+A')/2;
            % It seems that it is important to keep B to be positive definite.
            B = rand(N,N)*0.01;
            B = (B+B')/2;
            %B = B - eigs(B,1,'smallestreal')*eye(N);
            B = B - min(eig(B))*eye(N);
            B = -B;
            
            % parameters for ASQN
            opts.hess = @hess;
            opts.grad = @grad;
            opts.record = 1;
            opts.xtol = 0;1e-6;
            opts.ftol = 0;
            opts.gtol = 1e-10;
            opts.maxit = 200;
            opts.fun_extra = @fun_extra;
            
            opts.opts_init.record = 1;
            opts.solver_init = [];
            opts.opts_init.tau   = 1e-3;
            opts.opts_init.maxit = 2000;
            opts.opts_init.gtol  = opts.gtol*1e3;
            opts.opts_init.xtol  = opts.xtol*1e2;
            opts.opts_init.ftol  = opts.ftol*1e2;
            opts.opts_sub.record = 0;
            opts.opts_sub.tau    = 1e-3;
            opts.opts_sub.maxit  = [100,150,200,300,500];
            
            opts.opts_sub.gtol   = opts.gtol*1e0;
            opts.opts_sub.xtol   = opts.xtol*1e0;
            opts.opts_sub.ftol   = opts.ftol*1e0;
            opts.fun_TR = [];
            opts.tau = 1e-10;
            opts.theta = 1;
            
            % eigs
            [X0,~] = qr(randn(N,p),0);
            t0= tic; Fun = @(x) AFun(x) + BFun(x);
            nAx = 0; countA = 0; px = 0; countB = 0; timeB = 0;
            opts_eigs.issym = 1; opts_eigs.tol = 1e-10;
            [XExact,LamExact] = eigs(Fun, N, p, 'SA', opts_eigs);
            LamExact = diag(LamExact); t_eigs = toc(t0);
            fprintf('time of eigs %.2f, BV %.2f \n', t_eigs, timeB);

            % iter. info. of eigs 
            BX = Fun(XExact); XtG = XExact'*BX; G = BX - XExact*XtG;
            out_eigs.nrmG = max(sqrt(sum(G.*G,1)')./max(1,abs(diag(XtG))));
            out_eigs.time = t_eigs; out_eigs.timeB = timeB;
            out_eigs.nAx = nAx; out_eigs.countA = countA;
            out_eigs.px = px; out_eigs.countB = countB;
            
            % LOBPCG
            tol = 1e-10; nAx = 0; countA = 0; px = 0; countB = 0; timeB = 0;
            [XExact,LamExact,~,~,residualNorm] = lobpcg(X0, Fun, tol, 1000);
            LamExact = diag(LamExact); t_lobpcg = toc(t0); 
            fprintf('time of lobpcg %.2f, BV %.2f \n', t_lobpcg, timeB);

            % iter. info. of LOBPCG
            BX = Fun(XExact); XtG = XExact'*BX; G = BX - XExact*XtG;
            nrmG = max(sqrt(sum(G.*G,1)')./max(1,abs(diag(XtG))));
            out_lobpcg.nAx = nAx; out_lobpcg.countA = countA;
            out_lobpcg.px = nAx; out_lobpcg.countB = countB;
            out_lobpcg.res = max(residualNorm(:,end)); 
            out_lobpcg.time = t_lobpcg; out_lobpcg.timeB = timeB;
            
            % ASQN with augmented subspace {X^{k-1}, X^k}
            name = strcat('N-',num2str(N),'-p-',num2str(p));
            opts.usenystrom = 1; nAx = 0; countA = 0; px = 0; countB = 0; timeB =0;
            t0 = tic; [X,~,out_qnace] = asqn_eig(X0, @AFun, @BFun, p, opts); t_qnace = toc(t0);
            out_qnace.err = norm((A+B)*X - X*X'*(A+B)*X, 'fro');
            out_qnace.time = t_qnace; out_qnace.timeB = timeB;
            out_qnace.nAx = nAx; out_qnace.countA = countA;
            out_qnace.px = px; out_qnace.countB = countB;
            
            % ASQN with subspace {X^k} (not augmented)
            opts.usenystrom = 0; nAx = 0; countA = 0; px = 0; countB = 0; timeB = 0;
            t0 = tic; [X,~,out_ace] = asqn_eig(X0, @AFun, @BFun, p, opts); t_ace = toc(t0);
            out_ace.err = norm((A+B)*X - X*X'*(A+B)*X, 'fro'); 
            out_ace.nAx = nAx; out_ace.countA = countA; 
            out_ace.px = px; out_ace.countB = countB;
            out_ace.time = t_ace; out_ace.timeB = timeB;
            
            fprintf('----(N,p) = (%d,%d)-----\n',N,p);
            fprintf('eigs: #Ax/Ax/#BxBx: %.0f/%.0f/%.0f/%.0f \t time: %.2f \t timeB: %.2f \t err(2-norm of eig. val.): %.2e \n', out_eigs.nAx, out_eigs.countA,out_eigs.px, out_eigs.countB, t_eigs, out_eigs.timeB, out_eigs.nrmG);
            fprintf('LOBPCG: #Ax/Ax/#Bx/Bx: %.0f/%.0f/%.0f/%.0f \t time: %.2f \t timeB: %.2f \t err(2-norm of eig. val.): %.2e \n', out_lobpcg.nAx, out_lobpcg.countA,out_lobpcg.px, out_lobpcg.countB, t_lobpcg, out_lobpcg.timeB, max(residualNorm(:,end)));
            fprintf('ASQN: #Ax/Ax/#Bx/Bx: %.0f/%.0f/%.0f/%.0f \t time: %.2f \t timeB: %.2f \t err(2-norm of eig. val.): %.2e \n', out_qnace.nAx,out_qnace.countA,out_qnace.px,out_qnace.countB, t_qnace, out_qnace.timeB, out_qnace.nrmG);
            fprintf(' ACE: #Ax/Ax/#Bx/Bx: %.0f/%.0f/%.0f/%.0f \t time: %.2f \t timeB: %.2f \t err(2-norm of eig. val.): %.2e \n', out_ace.nAx, out_ace.countA, out_ace.px, out_ace.countB, t_ace, out_ace.timeB, out_ace.nrmG);
            fprintf(fid, '----(N,p) = (%d,%d)-----\n',N,p);
            fprintf(fid, 'eigs: #Ax/Ax/#Bx/Bx: %.0f/%.0f/%.0f/%.0f \t time: %.2f \t timeB: %.2f \t err(2-norm of eig. val.): %.2e \n', out_eigs.nAx, out_eigs.countA,out_eigs.px, out_eigs.countB, t_eigs, out_eigs.timeB, out_eigs.nrmG);
            fprintf(fid, 'LOBPCG: #Ax/Ax/#Bx/Bx: %.0f/%.0f/%.0f/%.0f \t time: %.2f \t timeB: %.2f \t err(2-norm of eig. val.): %.2e \n', out_lobpcg.nAx, out_lobpcg.countA,out_lobpcg.px, out_lobpcg.countB, t_lobpcg, out_lobpcg.timeB, max(residualNorm(:,end)));
            fprintf(fid, '  ASQN: #Ax/Ax/#Bx/Bx: %.0f/%.0f/%.0f/%.0f \t time: %.2f \t timeB: %.2f \t err(2-norm of eig. val.): %.2e \n', out_qnace.nAx,out_qnace.countA,out_qnace.px,out_qnace.countB, t_qnace, out_qnace.timeB, out_qnace.nrmG);
            fprintf(fid, '   ACE: #Ax/Ax/#Bx/Bx: %.0f/%.0f/%.0f/%.0f \t time: %.2f \t timeB: %.2f \t err(2-norm of eig. val.): %.2e \n', out_ace.nAx, out_ace.countA, out_ace.px, out_ace.countB, t_ace, out_ace.timeB, out_ace.nrmG);
            
            save(strcat('./results', filesep,'eigs-exB-rand-new-', name), 'out_eigs');
            save(strcat('./results', filesep,'lobpcg-exB-rand-new-', name), 'out_lobpcg');
            save(strcat('./results', filesep,'qnace-exB-rand-new-', name), 'out_qnace');
            save(strcat('./results', filesep, 'ace-exB-rand-new-', name), 'out_ace');
            
        end
    end
end

    function y = BFun(x)
        tstart = tic;
        px = px + size(x,2);
        countB = countB + 1;
        y = zeros(size(x));
        for i = 1:19
            y = y + B*x;
        end
        y = y / 19;
        timeB = timeB + toc(tstart);
    end

    function y = AFun(x)
        nAx = nAx + size(x,2);
        countA = countA + 1;
        y = A*x;
    end
end


function [X, G, out] = asqn_eig(X, A, BFun, p, opts, varargin)
% structured quasi-Newton method for lipar eigenvalue problem
%   min 1/2 tr(X'(A+BFun)X)   s.t. X'X = I_p 
%
% Input:
%         X --- initial guess
%         A --- ``cheap'' lipar operator
%      BFun --- ``expansive'' lipar operator
%         p --- number of eigenpairs   
%      opts --- options structure with fields
%               record = 0, no print out
%               maxit  max number of iterations
%               xtol   stop control for ||X_k - X_{k-1}||
%               gtol   stop control for the projected gradient
%               ftol   stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%               gamma1, gamma2, gamma3 and gamma4 parameter for adjusting
%               the regularization parameter
%               tau    initial value of regularization parameter
%               solver_init solver for obtaining a good intial guess
%               opts_init options structure for initial solver with fields
%                        record = 0, no print out
%                        maxit  max number of iterations
%                        xtol   stop control for ||X_k - X_{k-1}||
%                        gtol   stop control for the projected gradient
%                        ftol   stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%               solver_sub  solver for subproblem, LOBPCG is used here
%               opts_sub    options structure for subproblem solver with fields
%                        record = 0, no print out
%                        maxit  max number of iterations
%                        xtol   stop control for ||X_k - X_{k-1}||
%                        gtol   stop control for the projected gradient or
%                               the accuracy for solving the pwton direction
%                        ftol   stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                               usually, max{xtol, gtol} > ftol
%
% Output:
%         x --- solution
%         G --- gradient at x
%       out --- output information
% -----------------------------------------------------------------------
% Reference:
%  J. Hu, B. Jiang, L. Lin, Z. Wen and Y. Yuan
%  Structured Quasi-Newton Methods for Optimization with 
%  Orthogonality Constraints
% 
%   Author: J. Hu, Z. Wen
%  Version 1.0 .... 2018/9


%------------------------------------------------------------------------

if nargin < 3
    error('at least for inputs: [x, G, out] = arnt(x, A, BFun, p)');
elseif nargin < 4
    opts = [];
end

%-------------------------------------------------------------------------
% options for the trust region solver
if ~isfield(opts, 'gtol');           opts.gtol = 1e-6;  end % 1e-5
if ~isfield(opts, 'xtol');           opts.xtol = 1e-9;  end
if ~isfield(opts, 'ftol');           opts.ftol = 1e-16; end % 1e-13

if ~isfield(opts, 'eta1');           opts.eta1 = 1e-2;  end
if ~isfield(opts, 'eta2');           opts.eta2 = 0.9;   end
if ~isfield(opts, 'gamma1');         opts.gamma1 = 0.2; end
if ~isfield(opts, 'gamma2');         opts.gamma2 = 1;   end
if ~isfield(opts, 'gamma3');         opts.gamma3 = 1e1;  end
if ~isfield(opts, 'gamma4');         opts.gamma4 = 1e2;  end

if ~isfield(opts, 'maxit');          opts.maxit = 200;  end
if ~isfield(opts, 'record');         opts.record = 0;   end
if ~isfield(opts, 'model');          opts.model = 1;    end

if ~isfield(opts, 'eps');            opts.eps = 1e-14;  end
if ~isfield(opts, 'tau');            opts.tau = 10;     end
if ~isfield(opts, 'kappa');          opts.kappa = 0.1;  end
if ~isfield(opts, 'usenumstab');     opts.usenumstab = 1;  end

if ~isfield(opts, 'solver_sub');  opts.solver_sub = @RGBB;   end

hasRecordFile = 0;
if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
end


%--------------------------------------------------------------------------
% copy parameters
xtol    = opts.xtol;    gtol   = opts.gtol;    ftol   = opts.ftol;
eta1    = opts.eta1;    eta2   = opts.eta2;    usenumstab = opts.usenumstab;
gamma1  = opts.gamma1;  gamma2 = opts.gamma2;  gamma3 = opts.gamma3;
gamma4  = opts.gamma4;  maxit = opts.maxit;   record = opts.record;
eps   = opts.eps;       tau = opts.tau;    kappa  = opts.kappa;

%--------------------------------------------------------------------------
% GBB for Good init-data
opts_init = opts.opts_init;
% opts_init = opts.opts_init;
solver_init = opts.solver_init;
% -------------------------------------------------------------------------

% out.nfe = 1;
timetic = tic();
% ------------
% Initialize solution and companion measures: f(x), fgrad(x)

if ~isempty(solver_init)
    
    t1 = tic; [X,~, outin] = feval(solver_init, X, A, BFun, p, opts_init); t2 = toc(t1);

    % iter. info. 
    init = outin.iter;
    out.nfe = outin.nfe + 1;
    out.x0 = X;
    out.intime = t2;
else
    out.x0 = X; out.nfe = 1; out.intime = 0;
end

out.x0 = X; out.nfe = 1; out.intime = 0;
% compute function value and Euclidean gradient
BX = BFun(X); ABX = A(X) + BX; Numcount = 1;
F = iprod(X,ABX); [~, N] = size(X);

% Riemannian gradient
[XtG, G] = projection(X, ABX);
nrmG = max(sum(G.*G,1)'./max(1,diag(XtG)));

Xp = X; Fp = F; Gp = G; BXp = BX;

%------------------------------------------------------------------
% OptM for subproblems in TR
opts_sub.tau   = opts.opts_sub.tau;
opts_sub.gtol  = opts.opts_sub.gtol;
opts_sub.xtol  = opts.opts_sub.xtol;
opts_sub.ftol  = opts.opts_sub.ftol;
opts_sub.record = opts.opts_sub.record;
stagnate_check = 0;

if record
    str1 = '    %6s';
    stra = ['%8s','%13s  ',str1,str1,str1,str1,str1,str1,'\n'];
    str_head = sprintf(stra,...
        'iter', 'F', 'nrmG', 'XDiff', 'FDiff', 'mDiff', 'ratio', 'tau');
    str1 = '  %3.2e';
    str_num = ['(%3d,%3d)','  %14.8e', str1,str1,str1,str1,str1,str1,'\n'];
end

if hasRecordFile
    fprintf(fid,stra, ...
        'iter', 'F', 'nrmG', 'XDiff', 'FDiff', 'mDiff', 'ratio', 'tau');
end

V = X; W = BX;
% main loop
for iter = 1:maxit
    
    % set the no. of maximal iter. and stagnate check
    if nrmG >= 1
        opts_sub.maxit = opts.opts_sub.maxit(1);
        stagnate_check = max(stagnate_check,10);
    elseif nrmG >= 1e-2
        opts_sub.maxit = opts.opts_sub.maxit(2);
        stagnate_check = max(stagnate_check,20);
    elseif nrmG >= 1e-3
        opts_sub.maxit = opts.opts_sub.maxit(3);
        stagnate_check = max(stagnate_check,50);
    elseif nrmG >= 1e-4
        opts_sub.maxit = opts.opts_sub.maxit(4);
        stagnate_check = max(stagnate_check,80);
    else
        opts_sub.maxit = opts.opts_sub.maxit(5);
        stagnate_check = max(stagnate_check,100);
    end
    opts_sub.stagnate_check = stagnate_check;
    
    % criterion of PCG
    opts_sub.gtol = min(0.1*nrmG, 0.1);
    rreg = 10 * max(1, abs(Fp)) * eps; % safeguarding
    if usenumstab
        opts_sub.rreg = rreg;
    else
        opts_sub.rreg = 0;
    end
    
    % subproblem solving
    
    % store the information of current iteration
    TrRho = tau; % remark: ...
    sigma = tau*nrmG;
    data.TrRho = TrRho;
    data.XP = Xp;
    data.sigma = tau*nrmG; % regularization parameter
    
    % solve the subproblem contsructed by structured quasi-Newton 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    C = pinv(full(W'*V),1e-6);
    SubFun = @(x) A(x) + W*(C*(W'*x)) - sigma*(Xp*(Xp'*x));
    tol = max(gtol,min(1e-3, 0.1*nrmG)); % tol = 1e-10;
    [X, Lam] = lobpcg(Xp, SubFun,tol,1000);
    out_sub.fval = sum(Lam) - Fp + sigma*p;
    out_sub.iter = 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %------------------------------------------------------------------
    % compute fucntion value and Riemannian gradient
    BX = BFun(X); ABX = A(X) + BX; F = iprod(X, ABX);
    [G,XtG] = projection(X, ABX);
    Numcount = Numcount + 1;
    
    out.nfe = out.nfe + 1;
    out.nfe_sub = 1;
    out.iter_sub = 1;
    
    % compute the real and predicted reduction
    redf  = Fp - F + rreg;
    mdiff = - out_sub.fval + rreg;
    
    % compute the ration
    model_decreased = mdiff > 0;
    if ~model_decreased % if the model didn't decrase, ratio = -1
        if record; fprintf('model did not decrease\n'); end
        ratio = -1;
    else
        if (abs(redf)<=eps && abs(mdiff)<= eps) || redf == mdiff
            ratio = 1;
        else
            ratio = redf/mdiff;
        end
    end
    
    XDiff = norm(X-Xp,'inf');
    FDiff = abs(redf)/(abs(Fp)+1);
    
    % taken from LMSVD
    % Xin Liu, Zaiwen Wen and Yin Zhang, Limited Memory Block Krylov 
    % Subspace Optimization for Computing Dominant Singular Value 
    % Decompositions, SIAM Journal on Scientific Computing, 35-3 (2013), 
    % A1641-A1668.
    if opts.usenystrom
        % V = orth([X,Xp]); W = BFun(V); %[BX, BXp];
        XXp = X'*Xp;
        Px = Xp - X*XXp; Py = BXp - BX*XXp;
        T = Px'*Px; dT = diag(T);
        L = size(T,1);
        [sdT,idx] = sort(dT,'descend');
        L = sum(sqrt(sdT) > 1e-15,1);
        Icut = idx(1:L); Px = Px(:,Icut); 
        Py = Py(:,Icut); T = T(Icut,Icut);
        % V = [X, Xp]; W = [BX, BXp];
        % orthonormalize Px
        [U,D] = eig(T);
        ev = diag(D);
        [~,idx] = sort(ev,'ascend');
        e_tol = min(sqrt(eps),1e-13);
        cut = find(sqrt(ev(idx)) > e_tol,1);
        Icut = idx(cut:end);
        L = L - cut + 1;
        dv = 1./sqrt(ev(idx(Icut)));
        T = U(:,Icut)*sparse(1:L,1:L,dv);
        % subspace optimization
        V = [X, Px*T];
        W = [BX, Py*T];
    end
    % accept X
    if ratio >= eta1 && model_decreased
        Xp = X;  Fp = F; Gp = G; BXp = BX;
        nrmG = max(sqrt(sum(G.*G,1)')./max(1,abs(diag(XtG))));
    end
    
    if ~opts.usenystrom
        V = Xp; W = BXp;
    end
    
    out.nrmGvec(iter) = nrmG;
    out.fvec(iter) = F;
    
    % ---- record ----
    if record
        if iter == 1
            fprintf('switch to ASQN method \n');
            fprintf('%s', str_head);
        end
        fprintf(str_num, ...
            iter, out_sub.iter, Fp, nrmG, XDiff, FDiff, mdiff, ratio, tau);
    end
    
    if hasRecordFile
        fprintf(fid, str_num, ...
            iter, out_sub.iter, Fp, nrmG, XDiff, FDiff, mdiff, ratio, tau);
    end
    
    % ---- termination ----
    if nrmG <= gtol || ( (FDiff <= ftol) && ratio > 0 )
        out.msg = 'optimal';
        if nrmG  < gtol, out.msg = strcat(out.msg,'_g'); end
        if FDiff < ftol, out.msg = strcat(out.msg,'_f'); end
        break;
    end
    
    % update regularization parameter
    if ratio >= eta2
        tau = max(gamma1*tau, 1e-13);
        %         tau = gamma1*tau;
    elseif ratio >= eta1
        tau = gamma2*tau;
    elseif ratio > 0
        tau = gamma3*tau;
    else
        tau = gamma4*tau;
    end
    
    % if pgative curvature was encoutered, future update regularization parameter
    %     if out_sub.flag == -1
    %         tau = max(tau, out_sub.tau/nrmG + 1e-4);
    %     end
    
    
end % end outer loop
timetoc = toc(timetic);

% store the iter. no.
out.XDiff = XDiff;
out.FDiff = FDiff;
out.nrmG = nrmG;
out.fval = F;
out.iter = iter;
out.nfe = out.nfe + sum(out.nfe_sub);
out.time = timetoc;
out.Numcount = Numcount;

    function z = iprod(x,y)
        z = real(sum(sum(x.*y)));
    end

    function [G, XtG] = projection(X,G)
        XtG = X'*G;
        G = G - X*XtG;
    end

end




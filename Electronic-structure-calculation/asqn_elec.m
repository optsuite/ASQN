function [x, G, out] = asqn_elec(mol, opts, varargin)
% structured quasi-Newton method for electronic structure calculation
%   min E(X)   s.t. X^*X = I
%
% Input:
%       mol --- molecule
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
%               solver_sub  solver for subproblem
%               opts_sub    options structure for subproblem solver with fields
%                        record = 0, no print out
%                        maxit  max number of iterations
%                        xtol   stop control for ||X_k - X_{k-1}||
%                        gtol   stop control for the projected gradient or
%                               the accuracy for solving the Newton direction
%                        ftol   stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                               usually, max{xtol, gtol} > ftol
%                        hess   Euclidean Hessian
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

if nargin < 1
    error('at least one inputs: [x, G, out] = asqn(mol)');
elseif nargin < 2
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
if ~isfield(opts, 'usehse');         opts.usehse = 0;  end
if ~isfield(opts, 'useqnks');        opts.useqnks = 0;  end
if ~isfield(opts, 'usepartks');      opts.usepartks = 0;  end

if ~isfield(opts, 'solver_sub');  opts.solver_sub = @RGBB;   end

hasRecordFile = 0;
if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
end


%--------------------------------------------------------------------------
% copy parameters
xtol    = opts.xtol;    gtol    = opts.gtol;    ftol   = opts.ftol;
eta1    = opts.eta1;    eta2    = opts.eta2;    usenumstab = opts.usenumstab;
gamma1  = opts.gamma1;  gamma2  = opts.gamma2;  gamma3 = opts.gamma3;
gamma4  = opts.gamma4;  maxit   = opts.maxit;    record = opts.record;
eps     = opts.eps;       tau   = opts.tau;        kappa  = opts.kappa;
usehse  = opts.usehse;  useqnks = opts.useqnks; usepartks = opts.usepartks;
solver_sub  = opts.solver_sub;

%---- If no initial point x is given by the user, generate one at random.
if (nargin < 3)
    % use default opts
    optsKS = setksopt;
end

% verbose = 0; optsKS.verbose = 'off';
verbose    = ~strcmpi(optsKS.verbose,'off');
maxcgiter  = optsKS.maxcgiter;
cgtol      = optsKS.cgtol;
x          = optsKS.X0;
rho        = optsKS.rho0;

nspin = mol.nspin;
nocc  = mol.nel/2*nspin;
n1 = mol.n1; n2 = mol.n2; n3 = mol.n3;

if isfield(opts, 'X');  x = opts.X;  end

if isfield(opts, 'H')
    H = opts.H;
else
    if (isempty(rho))
        H = Ham(mol);
    else
        H = Ham(mol,rho);
    end
end

% generate initial guess of the wavefunction if it is not provided
if ( isempty(x) )
    x = genX0(mol,nocc);
    initX = 0;
else
    % check the dimension of X and make sure it is OK
    initX = 1;
    if ( x.ncols < nocc )
        fprintf('Error: The number of columns in X is less than nocc\n');
        fprintf('Error: size(X,2) = %d, nocc = %d\n', get(X,'ncols'), nocc);
        return;
    end;
    m1 = mol.n1; m2 = mol.n2; m3 = mol.n3;
    if (m1 ~= n1 | m2 ~= n2 | m3 ~= n3)
        error('dcm: dimension of the molecule does not match that of the wavefunction')
    end;
end;

% size information
nxcols = x.ncols;

% extract the ionic and external potential for possible reuse,
% save a copy of total potential for self-consistency check
vion = H.vion;
vext = H.vext; % bhlee

%% calculate Ewald and Ealphat (one time calculation)
Ewald     = getEewald(mol);
Ealphat   = getEalphat(mol);

% construct a preconditioner for LOBCG
prec = genprec(H);
if ~initX %(~initX || ~initerho)
    % run a few LOBPCG iterations to get a better set of wavefunctions
    [X, ev] = updateX(mol, H, x, prec, optsKS);
end

% exchange operator
x.occ = ones(1,x.ncols);

% length of memory
mm = 10; 

% initial settings for quasi-Newton method
istore = 0;  perm = []; ppos = 0; pp1 = zeros(mm,1); pp2 = pp1; 
alpha = 1e-3;

STY = zeros(mm);
STBS = zeros(mm);

% dimensions of variable $X$
[n,m] = size(x.psi);

%--------------------------------------------------------------------------
% GBB for Good init-data
opts_init = opts.opts_init;
% opts_init = opts.opts_init;
solver_init = opts.solver_init;
% -------------------------------------------------------------------------

timetic = tic();
% ------------
% Initialize solution and companion measures: f(x), fgrad(x)

if ~isempty(solver_init)
    opts_init.gtol  = opts.gtol*1e3;
    opts_init.xtol  = opts.xtol*1e2;
    opts_init.ftol  = opts.ftol*1e2;
    tic; [~, x, ~, ~, ~, outin] = GBB_DFT(mol, opts_init);
    H = outin.H; 

    
    init = outin.iter;
    out.nfe = outin.nfe + 1;
    out.x0 = x;
    out.intime = outin.time;
else
    out.x0 = x; out.nfe = 1; out.intime = 0;
end

% compute function value and Euclidean gradient
fun(x);

if useqnks
    if usehse
        Initcomp = calculateACE2(VexxX, x);
        VexxX_old = VexxX; Vexx2_old = Vexx2;
    else
        Initcomp = @(x) 0*x;
    end
else
    if usehse
        Initcomp = calculateACE2(VexxX, x);
        VexxX_old = VexxX; Vexx2_old = Vexx2;
    end
end

% Riemannian gradient
G = projection(x,Ge);  nrmG  = norm(G, 'fro');
out.fval_init = F; out.nrmG_init = nrmG;

xp = x; Fp = F; Gp = G; Gep = Ge; Hp = H; % Vexx2_old = Vexx2;
Hcomp = @(x) (H*x)*(4/mol.nspin);% Vexx2_old = Vexx2;
% H0 = H; H0.vtot = vion + vext;

%------------------------------------------------------------------
% OptM for subproblems in TR
opts_sub.tau   = opts.opts_sub.tau;
opts_sub.gtol  = opts.opts_sub.gtol;
opts_sub.xtol  = opts.opts_sub.xtol;
opts_sub.ftol  = opts.opts_sub.ftol;
opts_sub.record = opts.opts_sub.record;
if isfield(opts.opts_sub, 'recordFile')
    opts_sub.recordFile = opts.opts_sub.recordFile;
end
stagnate_check = 0; s1 = 0; s2 = 0;

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

% main loop
fprintf('switch to ASQN method \n');
fprintf('%s', str_head);

opts_sub.stagnate = 0; z = x;
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
    opts_sub.alpha = alpha;
    
    % subproblem solving
    
    % store the information of current iteration
    sigma = tau*nrmG; % regularization parameter
    
    % solve the subproblem
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~useqnks % whether to split the Hessian of KS part 
        % store the unvaried information in inner iteration
        hvx = fun_extra(xp);
        if usehse % whether to add the Hessian of Fock energy part, KS or HF?
            Ehess =  @(U) hess(xp,U) + qnb(U);
        else
            Ehess =  @(U) hess(xp,U);
        end
        
    else
        Ehess =  @(U) qnb(U);
    end
        
    % stagnate check 
    if iter > 12        
        res = out.nrmGvec(iter-11:iter-2) ./ out.nrmGvec(iter-10:iter-1);
        if max(res) < 1.1 && min(res) > 0.9
            Ehess =  @(U) (sigma + 1)*U;
            opts_sub.stagnate = 1;
        end
    end
    
    % Subspace refinement
    opts_gbb.maxit = 40;
    opts_gbb.gtol  = 1e-6;
    opts_gbb.xtol  = 0;
    opts_gbb.ftol  = 0;
    opts_gbb.H = Hp; opts_gbb.X0 = xp;
    opts_gbb.usehse = usehse;
    if opts_sub.stagnate
        [~,~, x, out_sub] = sDFT(mol, opts_gbb);
        out_sub.flag = 1; out_sub.fval0 = Fp;
        
        s1 = s1 + out_sub.iter-1;
        sub_iter_tot = sum(out_sub.iter_sub(2:end));
        % nfe_iter_tot = sum(nfe_sub.nfe_sub(2:end));
        s2 = s2 + sub_iter_tot;
        
        % store the iter. info. of inner iter.
        out_sub.iter = out_sub.iter_sub(1);
        out_sub.nfe = out_sub.nfe_sub(1);
        out_sub.fval0 = Fp;
        
           
        out.iter_sub(iter) = out_sub.iter; % inner iteration no.
        out.nfe_sub(iter)  = out_sub.nfe; % ehess-matrix product and retraction no.

        opts_sub.stagnate = 0;

    else
        if strcmp(func2str(solver_sub), 'RNewton_DFT')
            Ehess = @(U) Ehess(U) + sigma*U;
            [x, out_sub]= feval(solver_sub, xp, Gep, Gp, Ehess, opts_sub);
        else
            [x, out_sub] = feval(solver_sub, mol, xp, Ehess, Gep, Gp, sigma, opts_sub);
        end
        
        % store the iter. info. of inner iter.
        out.iter_sub(iter) = out_sub.iter; % inner iteration no.
        out.nfe_sub(iter)  = out_sub.nfe; % ehess-matrix product and retraction no.
        out.flag(iter) = out_sub.flag;
        out.time_sub(iter) = toc(timetic);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %------------------------------------------------------------------
    % compute fucntion value and Riemannian gradient
    fun(x);
    G = projection(x,Ge);
    
    out.nfe = out.nfe + 1;
    
    % compute the real and predicted reduction
    redf = Fp - F + rreg;
    mdiff = out_sub.fval0 - out_sub.fval + rreg;
    
    % compute the ratio
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
    
    XDiff = norm(x-xp,'inf');
    FDiff = abs(redf)/(abs(Fp)+1);
    s = x - xp; x_tmp = x;
    Y = G - Gp; SY = abs(iprod(s,Y));
    if mod(iter,2)==0; alpha = (norm(s,'fro')^2)/SY;
    else alpha = SY/(norm(Y,'fro')^2); end
    
    % difference of the gradient 
    if useqnks; y1 = Ge - Gep;  end
    if usehse;  y2 = VexxX - VexxX_old; end
    if usehse || useqnks
        qnbs = qnb(s);
    end
    
    % accept X
    success = ratio >= eta1 && model_decreased;
    if success
        if usehse; VexxX_old = VexxX; Vexx2_old = Vexx2; end
        x_tmp = xp; xp = x;  Fp = F; Gp = G; Gep = Ge; Hp = H;
        nrmG = norm(Gp, 'fro'); opts_sub.usezero = 1;
    else
        opts_sub.usezero = 0; % opts_sub.deta = out_sub.deta;
    end
    out.nrmGvec(iter) = nrmG;
    out.fvec(iter) = F;
        
    % ---- record ----
    if record
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
    
    % quasi-Newton construction
    success = 1;
    newy = 0;
    if success
        if useqnks
            
            newy = 1;
            if usepartks % split the Hessian of KS part
                Hs = applyLinH(Hp,s)*(4/mol.nspin);
                Hcomp = @(x) applyLinH(Hp,x)*(4/mol.nspin);
            else
                Hs = Hp*s*(4/mol.nspin);
                Hcomp = @(x) (Hp*x)*(4/mol.nspin);
            end
            ygk = y1 - Hs;
            if usehse
                % Nystrom approximation
                if iter <= 10
                    Initcomp = calculateACE2(VexxX_old, xp);
                else
                    Initcomp = calculateACEm(VexxX_old, Vexx2_old(x_tmp),... 
                    xp, x_tmp);
                end
                
                % Remaining parts are initially approximated using BB-type
                % strategy
                ys = ygk - y2;
                sig = real(iprod(ys,s)/iprod(s,s));
                Initcomp = @(x) Initcomp(x) + sig*x;
            else
                y = ygk;
                sig = norm(y,'fro')/norm(s,'fro');
                Initcomp = @(x) sig*x;
            end
        else
            if usehse
                ygk = y2;
                Initcomp = calculateACE2(VexxX_old, xp);
                % Initcomp = calculateACEm(VexxX_old, Vexx2_old(x_tmp), xp, x_tmp);
                %                 if iter <= 10
                %                     Initcomp = calculateACE2(VexxX_old, xp);
                %                 else
                %                     Initcomp = calculateACEm(VexxX_old, Vexx2_old(x_tmp), xp, x_tmp);
                %                 end
                newy = 1;
            end
        end
        if newy
            
            % store info. of graient
            nrms = norm(s,'fro');
            ygkmbsk = ygk - qnbs;
            nrmyps = norm(ygkmbsk, 'fro');
            stygk = real(iprod(s, ygkmbsk));
            if abs(stygk) > 1e-8*nrms*nrmyps;
                istore = istore + 1;
                pos = mod(istore, mm); if pos == 0; pos = mm; end; ppos = pos;
                YK(:,pos) = ygk.psi(:); SK(:,pos) = s.psi(:); s.psi = SK(:,pos);
                
                if usehse
                    for i = 1:min(istore,mm)
                        s.psi = reshape(SK(:,i),n,m);
                        Vexxs = Initcomp(s);
                        BSK(:,i) = Vexxs.psi(:);
                    end
                    s.psi = SK(:,pos);
                end
                
                
                if istore <= mm
                    perm = [perm, pos]; perm2 = perm;
                    % STY is S'Y, lower triangular
                    if usehse
                        STBS = real(SK(:, 1:istore)'*BSK(:, 1:istore));
                        STBS = (STBS + STBS')/2;
                    else
                        STBS(1:istore, istore) = real(SK(:, 1:istore)'*s.psi);
                        STBS(istore, 1:istore) = STBS(1:istore, istore);
                    end
                    
                    STY(istore, 1:istore) = real(YK(:,1:istore)'*s.psi);
                
                else
                    ppos = mm; perm = [perm(mm), perm(1:mm-1)];
                    perm2 = [perm2(2:mm), perm2(1)];
                    
                    STY(1:end-1, 1:end-1) = STY(2:end, 2:end);
                    
                    if usehse
                        STBS = real(SK(:,perm2)'*BSK(:,perm2));
                        STBS = (STBS + STBS')/2;
                    else
                        STBS(1:end-1, 1:end-1) = STBS(2:end, 2:end);
                        STBS(perm,end) = real(s.psi'*SK);
                        STBS(end,:) = STBS(:,end);
                    end
                    
                    % then update the last column or row
                    STY(end, perm) = real(s.psi'*YK);
                end
                DD = diag(diag(STY(1:ppos,1:ppos)));
                LL = tril(STY(1:ppos,1:ppos), -1);
                % the inverse may be modified to Cholesky factorization
                if usehse
                    BBinv = inv(DD + LL + LL' - STBS(1:ppos,1:ppos));
                else
                    BBinv = inv(DD + LL + LL' - sig*STBS(1:ppos,1:ppos));
                end
                
            end
        end
        
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
    
    tau = min(tau, 1e13);
    
    %     if ratio > 0
    %         if nrmG < 0.25/tau
    %             tau = 4*tau;
    %             %         tau = gamma1*tau;
    %         elseif nrmG < 0.75/tau
    %             tau = 1*tau;
    %         else
    %             tau = max(0.25*tau, 1e-8);
    %         end
    %     else
    %         tau = 4*tau;
    %     end
    
    % if negative curvature was encoutered, future update regularization parameter
    if out_sub.flag == -1
        tau = max(tau, out_sub.tau/nrmG + 1e-4);
    end
    
    
end % end outer loop
timetoc = toc(timetic);

% store the iter. no.
out.H = Hp;
out.XDiff = XDiff;
out.FDiff = FDiff;
out.nrmG = nrmG;
out.fval = Fp;
out.iter = length(out.iter_sub);
out.nfe = out.nfe + sum(out.nfe_sub);
if s1
    out.iter = out.iter + s1;
    out.avginit = (sum(out.iter_sub) + s2)/out.iter;
else
    out.avginit = sum(out.iter_sub)/out.iter;
end

out.time = timetoc;

    function fun(X)
        rho1 = getcharge(mol,X);
        % vol = get(mol,'vol');
        % Kinetic energy and some additional energy terms
        KX = applyKIEP(H,X); % not multiply with 2/nspin, hence scale it in Ekin
        Ekin = (2/mol.nspin)*iprod(X,KX);
        % Compute Hartree and exchange correlation energy and potential
        [vhart,vxc,uxc2,rho1] = getVhxc(mol,rho1);
        
        % Calculate the potential energy based on the new potential
        Ecoul = getEcoul(mol,abs(rho1),vhart);
        Exc   = getExc(mol,abs(rho1),uxc2);
        F = Ewald + Ealphat + Ekin + Ecoul + Exc;
        
        if usehse
            Vexx = getVexx(X, mol, 0);
            fock = getExx(X, Vexx, mol);
            
            F = F + fock;
        else
            VexxX = [];
        end
        F = real(F);
        
        % Gradient
        vout = getVtot(mol, vion, vext, vhart, vxc);
        H.vtot = vout;
        Ge = (H*X)*(4/mol.nspin);
        if usehse
            Vexx2 = @(U) Vexx(U)*(4*mol.vol^2/(mol.n1*mol.n2*mol.n3));
            VexxX = Vexx2(X);
            Ge = Ge + VexxX;
        else
            VexxX = [];
        end
        
    end

    function hvx = fun_extra(XP)
        
        rho1 = getcharge(mol,XP);
        %         [vhart,vxc,uxc2,rho1] = getVhxc(mol,rho1);
        %
        %         vout = getVtot(mol, vion, vext, vhart, vxc);
        %         H.vtot = vout;
        hvx = getvhxc2nd(rho1);
        
    end

% vion, vert, vhart and vxc are 3-d tensor, mol and U are Molecule.
    function h = hess(X, U)
        
        h1 = Hp*U*(4/mol.nspin);
        
        h2 = applyHvxprod(mol,Hp,hvx,X,U);
        
        h = h1 + h2;
        
    end

% L-SR1
    function h = qnb(U)
        if ~ppos
            h = Initcomp(U);
            if useqnks
                h = h + Hcomp(U);% (Hp*U)*(4/mol.nspin);
            end
        else
            u = U.psi(:);
            % the inverse may be modified to Cholesky factorization
            if usehse
                pp1 = real(BSK(:,perm2(1:ppos))'*u);
                pp2 = real(YK(:,perm2(1:ppos))'*u);
                pp = pp2 - pp1;
            else
                pp1 = real(SK(:,perm2(1:ppos))'*u);
                pp2 = real(YK(:,perm2(1:ppos))'*u);
                pp = pp2 - sig*pp1;
            end
            
            % pp = [sig*(SK(:,perm2)'*U); YK(:,perm2)'*v];
            BBpp = BBinv*pp;
            h = Initcomp(U);
            if useqnks
                h = h + Hcomp(U); %(Hp*U)*(4/mol.nspin);
            end
            
            if usehse
                YKmBSK = YK - BSK;
                h = h + reshape(YKmBSK(:,perm2)*BBpp,n,m);
            else
                YKmBSK = YK - sig*SK;
                h = h + reshape(YKmBSK(:,perm2)*BBpp,n,m);
            end
            
        end
        
    end


    function Vexxf = getVexx(X, mol, dfrank, exxgkk, F)
        %
        % Usage: Vexx = getVexx(X, F, exxgkk, mol)
        %
        % Purpose:
        %    Computes exact exchange operator
        %
        % Input:
        %    X  --- Wavefunction
        %    F  --- Fourier Transform
        %    dfrank --- Rank for density fitting
        %    exxgkk --- Eigenvalue of exchagne operator
        %    mol --- Molecule information
        %
        % Ouptut:
        %    Vexx --- Exact exchange operator
        %
        if nargin < 5, F = KSFFT(mol);  end
        if nargin < 4, exxgkk = getExxgkk(mol); end
        
        % n123 = mol.n1 * mol.n2 * mol.n3;
        Phi = F' * X.psi(:,logical(X.occ));
        if dfrank
            Vexxf = @(Psi)Vexxdf(Psi, Phi, dfrank, F, exxgkk, mol);
        else
            Vexxf = @(Psi)Vexx(Psi, Phi, F, exxgkk, mol);
        end
        
    end

    function G = projection(X,G)
        % projection onto the tangent space
        GX = X'*G; GX = (GX + GX')/2;
        G  = G - X*GX;
    end

    function Q = myQR(XX)
        [Q, R] = qr(XX, 0);
        Q = Q * diag(sign(sign(diag(R))+.5));
    end

end




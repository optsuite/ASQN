function [x, out] = RNewton_DFT(x0, egrad, rgrad, H, opts)
% modified Newton method to solve the subproblem
% min <egrad, x - x_k> + 1/2*<H[x - x_k], x - x_k> 
%   + 1/2*sigma_k \|x -  x_k\|^2
% 
% Input:
%         x0 --- initial guess
%      egrad --- Euclidean gradient of original function at x_k
%      rgrad --- Riemannian gradient of original function at x_k
%          H --- Euclidean Hessian of original function at x_k
%       opts --- option structure with fields
%                gtol         the accuracy for solving the Newton direction
%                eta, rhols   parameters in line search
%                maxit        max number of iterations  
%                record       = 0, no print out
%
% Output: 
%          x --- soultion
%        out --- output information
%
% -----------------------------------------------------------------------
% Reference: 
%  J. Hu, A. Milzark, Z. Wen and Y. Yuan
%  Adaptive Regularized Newton Method for Riemannian Optimization
%
% Author: J. Hu, Z. Wen
%  Version 1.0 .... 2017/8


% termination rule
if ~isfield(opts, 'gtol');       opts.gtol = 1e-7;   end % 1e-5
if ~isfield(opts, 'eta');        opts.eta  = .2;     end % 1e-5
if ~isfield(opts, 'rhols');      opts.rhols  = 1e-4;   end
if ~isfield(opts, 'maxit');      opts.maxit  = 200;    end
if ~isfield(opts, 'usezero');    opts.usezero = 1;      end
if ~isfield(opts, 'record');     opts.record = 0;      end
if ~isfield(opts, 'comp_eig');   opts.comp_eig = 0;      end
if ~isfield(opts, 'alpha');      opts.alpha = 1e-3;      end

% copy parameters
rhols = opts.rhols;  eta = opts.eta; gtol = opts.gtol; record = opts.record;

% numerical stability
if isfield(opts, 'rreg')
    rreg = opts.rreg;
else
    rreg = 0;
end

% record iter. info.
hasRecordFile = 0;
if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
end

% data structure of x0
if isstruct(x0)
    matX0 = x0.matX;
end

if record || hasRecordFile
    str1 = '  %6s';
    stra = ['%10s',str1,' %2s',str1,str1,str1,str1,str1,str1,str1,'\n'];
    str1 = '  %1.1e';
    str_num = ['%1.6e','  %d','      %d', str1,str1,str1,str1,'  %d',str1,'\n'];
    str = sprintf(stra,...
        'f: %1.6e', 'iters: %d', 'flag: %d', 'tol: %6s', 'geta: %6s', 'angle: %2.1f',...
        'err: %6s','pHp: %6s','nls: %d','step: %6s');
end


if hasRecordFile
    fprintf(fid,stra, ...
        'f', 'iters', 'flag', 'tol', 'geta','err','pHp','nls','step');
end

% set the initial iter. no.
out.nfe = 0; out.fval0 = 0;
% pcg; return -1 if the Hessian is indefinite
[deta,negcur,iters,err,pHp,resnrm,flag,mark,out_cg] ...
    = pcgw(x0,H,egrad,-rgrad,opts,gtol);

% record the iter. info.
out.nfe = out.nfe + iters;
out.iter = iters;
out.flag = flag;
out.tau = mark;
out.deta = deta;
out.red = out_cg.f_CG;

%%%%%%%%%%%%%%%%%%%%%%%%

% judge whether to use the negative direction
% If a not too small negative curvature is encoutered, 
% combine it to get a new direction
if mark > 1e-10
    gcur = iprod(rgrad,negcur);
    % negcur = -sign(gcur) * negcur;
    deta = deta + gcur/pHp * negcur;
end

% inner product between gradient and descent direction
geta = iprod(rgrad,deta); step = 1;
angle = acos(-geta/(norm(rgrad,'fro')*norm(deta,'fro')))*180/pi;
% if angle > 80
%     deta = -rgrad; geta = -norm(rgrad, 'fro')^2;
%     step = opts.alpha;
% end

% Armijo search with intial step size 1
nls = 1; deriv = rhols*geta; 
while 1
    x = myQR(x0 + step*deta);
    if isstruct(x)
        matX = x.U*x.S*x.V';
        x.matX = matX;
        xx0 = matX - matX0;
        Hxx0 = H(xx0);
    else
        xx0 = x - x0;
        Hxx0 = H(xx0);
    end
    
    f = iprod(egrad,xx0) + .5*iprod(Hxx0,xx0);
    out.nfe = out.nfe + 1;

    if f - rreg <= step*deriv || nls >= 5
        break;
    end
    step = eta*step;
    nls = nls + 1;
end

% print iter. info.
out.fval = f; opts.record = 1; out.d = deta; out.nls = nls;

if record
    fprintf(str, ...
        full(f), iters, flag, gtol, geta, angle, err, pHp, nls, step);  
end

if hasRecordFile
    fprintf(fid, str_num, ...
        full(f), iters, flag, gtol, geta, angle, err, pHp, nls, step);
end

end

function a = iprod(x,y)
a = real(sum(sum(conj(x).*y)));
%a = sum(sum(x.*y));
end


function  q = precondfun(r)

precond = 0;
%if isfield(par,'precond'); precond = par.precond; end

if (precond == 0)
    q = r;
    %    elseif (precond == 1)
    %       q = L.invdiagM.*r;
    %    elseif (precond == 2)
    %      if strcmp(L.matfct_options,'chol')
    %         q(L.perm,1) = mextriang(L.R, mextriang(L.R,r(L.perm),2) ,1);
    %      elseif strcmp(L.matfct_options,'spcholmatlab')
    %         q(L.perm,1) = mexbwsolve(L.Rt,mexfwsolve(L.R,r(L.perm,1)));
    %      end
    %      if isfield(par,'sig')
    %         q = q/par.sig;
    %      end
end
end

function [deta,negcur, iter,err,pHp,resnrm, flag,mark, out] ...
    = pcgw(x,H,grad,r,opts,tol)
if ~isfield(opts, 'maxit');  opts.maxit  = 200;   end
if ~isfield(opts, 'minit');  opts.minit  = 1;   end
if ~isfield(opts, 'stagnate_check');  opts.stagnate_check  = 50;   end
if ~isfield(opts, 'record'); opts.record  = 0;   end
if ~isfield(opts, 'usezero'); opts.usezero  = 1;   end

% copy parameters
maxit = opts.maxit;
minit = opts.minit;
stagnate_check = opts.stagnate_check;
record = 0; [n,m] = size(x.psi);
zero = zeros(n,m); % zero element in the tangent space
mark = 0; f_CG = 0;
alpha = 0; beta = 0;
comp_eig = opts.comp_eig;

% initial point
usezero = 1; record = 1;
if usezero
    deta = zero;
    Hdeta = zero;
else
    deta = opts.deta;
    Hdeta = H(deta);
    GX = x'*grad; GX = (GX  + GX')/2;
    pGX = p*GX; rHdeta = Hdeta - pGX;
    GX = x'*rHdeta; GX = (GX  + GX')/2;
    Hdeta = rHdeta - x*GX;
    r = r - Hdeta;
    deta = zero;
end

% set the initial iter. no.
r0 = r; z = precondfun(r);
p = z;
err = norm(r,'fro'); resnrm(1) = err; minres = err;
rho = iprod(r,z);
negcur = zero;
flag = 1; record = 0;

if record
    str1 = '   %6s';
    stra = ['%6s', str1, str1, str1,'\n'];
    str_head = sprintf(stra,...
        'iter', 'alpha', 'pHp', 'err');
    str1 = '   %1.2e';
    str_num = ['   %d', str1,str1,str1,'\n'];
end

% PCG loop
for iter = 1:maxit
    
    % Riemannian Hessian
    rHp = rhess(x,grad, H,p);
    pHp = iprod(p, rHp);
    nrmp = norm(p,'fro')^2;
    scalenrmp = 1e-10*nrmp;
    
    if comp_eig
        rhess_vec_test = @(p) rhess_vec(x,grad, H, p);
        opts_eigs.isreal = false; opts_eigs.issym = 1;
        eigs(rhess_vec_test, n*m, 10, 'LR', opts_eigs)
    end
    % check the stopping criterion, construct the new direction if stopped
    if pHp <= scalenrmp       
        if iter == 1; deta = p;
        else
            if pHp <= -scalenrmp
                negcur = p; mark = -pHp/nrmp;
            end
        end
        flag = -1; out.msg = 'negative curvature';
        f_CG_new = - iprod(deta,r0) + .5*iprod(deta,Hdeta);
        out.f_CG = f_CG_new;
        break;
    end
    
    if abs(pHp) < 1e-20
        
        out.msg = 'pHp is small';
        break;
    else
        alpha = rho/pHp;
        
        if iter == 1
            deta = alpha*p;
            Hdeta = alpha*rHp;
        else
            deta = deta + alpha*p;
            Hdeta = Hdeta + alpha*rHp;
        end
        f_CG_new = - iprod(deta,r0) + .5*iprod(deta,Hdeta);
        
        out.f_CG = f_CG_new;
        
        if f_CG_new > f_CG
            out.msg = 'no decrease in PCG';  
            fprintf('no decrease in PCG \n');
            break;
        else
            f_CG = f_CG_new;
        end
        r = r - alpha * rHp;
    end
    
    % residual
    err = norm(r,'fro'); resnrm(iter+1) = err;
    
    if record 
        if iter == 1
            fprintf('\n%s', str_head);
        end
        fprintf(str_num, ...
            iter, alpha, pHp, err);
    end
    
    % check stagnate and stopping criterion
    if (err < minres); minres = err; end
    if (err < tol) && (iter > minit);
        out.msg = 'accuracy'; break; end
    if (iter > stagnate_check) && (iter > 10)
        ratio = resnrm(iter-9:iter+1)./resnrm(iter-10:iter);
        if (min(ratio) > 0.97) && (max(ratio) < 1.03)
            flag = -2;
            out.msg = 'stagnate check';
            break;
        end
    end
    %%-----------------------------
    if (abs(rho) < 1e-16)
        flag = -3;
        out.msg = 'rho is small';
        break;
    else
        z = precondfun(r);
        rho_old = rho;
        rho = iprod(r,z);
        beta = rho/rho_old;
        p = z + beta*p;
    end
    
end
out.alpha = alpha;
out.beta = beta;
end

function Q = myQR(XX)
[Q, R] = qr(XX, 0);
Q = Q * diag(sign(sign(diag(R))+.5));
end

function rHp = rhess(x,grad, H, p)
Hp = H(p);
GX = x'*grad; GX = (GX  + GX')/2;
pGX = p*GX; rHp = Hp - pGX;
GX = x'*rHp; GX = (GX  + GX')/2;
rHp = rHp - x*GX;
end

function y = rhess_vec(x,grad, H, p)
[n,m] = size(x.psi);
p = reshape(p,n,m);
y = x; y.psi = p; p = y;
Hp = H(p);
GX = x'*grad; GX = (GX  + GX')/2;
pGX = p*GX; rHp = Hp - pGX;
GX = x'*rHp; GX = (GX  + GX')/2;
rHp = rHp - x*GX;
y = rHp.psi(:);

end


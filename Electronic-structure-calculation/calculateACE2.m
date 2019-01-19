function VexxACE = calculateACE2(Vexx, X)
%
% Usage: VexxACE = calculateACE(Vexx, X)
%
% Purpose:
%    Computes the Adaptively Compressed Exchange Operator
%
% Input:
%    Vexx --- Exact Exchange Operator
%    X --- Wavefunction
%
% Ouptut:
%    VexxACE --- ACE Exchange Operator
%

%%  linlin
%     W = Vexx(X);
%     M = X' * W;
%     M = (M + M')/2;
%     R = chol(-M);
%     Xi = W / R;
%     VexxACE = @(x) -Xi * (Xi' * x);
    
%% new
   W = Vexx;
   M = X'*W; M = (M + M')/2;
   C = pinv(M, 1e-6);
   VexxACE = @(x) W*(C*(W'*x));
     
end
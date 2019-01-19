function VexxACE = calculateACEm(Vexx1, Vexx2, X1, X2)
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
    
%% new hujiang
   % W = Vexx(X);
   W11 = X1'*Vexx1; W22 = X2'*Vexx2;
   W11 = (W11 + W11')/2; W22 = (W22 + W22')/2;
   W12 = X1'*Vexx2; W21 = W12';
   M = [W11, W12; W21, W22];
   W = [Vexx1, Vexx2];
   C = pinv(M, 1e-6);
   VexxACE = @(x) W*(C*(W'*x));
     
end
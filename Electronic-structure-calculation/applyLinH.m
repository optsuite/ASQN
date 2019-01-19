function HX = applyLinH(H,X)
% MTIMES  Overload multiplication operator for Ham class
%    HX = H*X returns a wavefun corresponding to H*X.
%
%    See also Ham, Wavefun.

%  Copyright (c) 2016-2017 Yingzhou Li and Chao Yang,
%                          Stanford University and Lawrence Berkeley
%                          National Laboratory
%  This file is distributed under the terms of the MIT License.

ncol = ncols(X);

% Apply Laplacian
KinX = repmat(H.gkin,1,ncol).*X;

% Apply nonlocal pseudopotential
VnlX = H.vnlmat*(repmat(H.vnlsign,1,ncol).*(H.vnlmat'*X));

HX = KinX + VnlX;

end
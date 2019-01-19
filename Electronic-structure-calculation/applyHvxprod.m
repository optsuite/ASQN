function HZ = applyHvxprod(mol,H,hxc,X,Z)

if ( isa(X,'Wavefun') && X.iscompact )
    %[n1,n2,n3] = size(H.vtot);
    n1 = X.n1;
    n2 = X.n2;
    n3 = X.n3;
    n123 = n1*n2*n3;
    vol = mol.vol;
    nspin = mol.nspin;

    idxnz = H.idxnz;
    ncols = X.ncols;
    
    Xpsi = X.psi;
    Zpsi = Z.psi;
    
    % ifftpsi = get(X,'ifftpsi');
    hasifft = 0;  % if ~isempty(ifftpsi); hasifft = 1; end
        
    XZ = zeros(n1,n2,n3);
    for j = 1:ncols
        if ~hasifft
            Xpsi3d = zeros(n1,n2,n3);
            %Xpsi3d(idxnz) = Xpsi{j};
            Xpsi3d(idxnz) = Xpsi(:,j);
            ifftpsi{j}    = ifftn(Xpsi3d);
        end
        Zpsi3d = zeros(n1,n2,n3);
        %Zpsi3d(idxnz) = Zpsi{j};
        Zpsi3d(idxnz) = Zpsi(:,j);
        iZpsi3d    = ifftn(Zpsi3d);
        XZ = XZ + (ifftpsi{j}).*conj(iZpsi3d); % conj is important
    end
    XZ = XZ*((16*n123^2)/(nspin^2*vol));
    
    %-----------------------
    %ecut2 = get(mol,'ecut2');
    %noshift = 1;
    %gmask2 = FreqMask(mol,ecut2,noshift);
    %idxnz2 = get(gmask2,'idxnz');
    %gkk2 = get(gmask2,'gkk');
    %gkk2 = get(H,'gkk2');
    %idxnz2 = get(H,'idxnz2');
    ecut2 = mol.ecut2;
    % add temperarily
    grid2 = Ggrid(mol,ecut2);
    idxnz2 = grid2.idxnz;
    gkk2 = grid2.gkk;
    
    rhog   = fftn(XZ);
    rhog   = rhog(idxnz2);
    w      = zeros(n1,n2,n3);
    inz    = find(abs(gkk2) ~= 0);
    w(idxnz2(inz)) = e2Def()*4*pi*rhog(inz)./gkk2(inz);
    hh     = real(ifftn(w));
    %------------------------
    
    % Calculate the exchange-correlation part
    hh  = hh + hxc.*XZ;    % real fun.   
%     hh = hxc.*XZ;
   
    for j = 1:ncols
        vXr3d  = hh.*ifftpsi{j};
        vX3d   = fftn(vXr3d);
        HZpsi{j} = vX3d(idxnz);
    end
    HZ = Wavefun(HZpsi,n1,n2,n3,idxnz);
else
    error('The Hamiltonian must operate on a Wavefun object');
end

function beta = iprod(X,Y)
%
% Perform pointwise multiplication of two Wavefun objects
%              Z = X.*Y;
%
if (nargin == 2)
    %
    % --- process the first operand ---
    %
    if ( isa(X,'Wavefun') )
        %
        % turn the cell array into a matrix
        %
        Xn123 = X.n1*X.n2*X.n3;
        if (X.iscompact)
            %Xmat = zeros(X.nrows,X.ncols);
            %for j = 1:X.ncols
            %    Xmat(:,j) = X.psi{j};
            %end;
            Xmat = X.psi;
        else
            Xmat = zeros(Xn123,X.ncols);
            for j = 1:X.ncols
                Xmat(:,j) = reshape(X.psi{j},Xn123,1);
            end;
        end
        %
        % --- process the second operand ---
        %
        if ( isa(Y,'Wavefun') )
            %
            % turn the cell array into a matrix
            %
            Yn123 = Y.n1*Y.n2*Y.n3;
            if (Y.iscompact)
                %Ymat = zeros(Y.nrows,Y.ncols);
                %for j = 1:Y.ncols
                %    Ymat(:,j) = Y.psi{j};
                %end;
                Ymat = Y.psi;
            else
                Ymat = zeros(Yn123,Y.ncols);
                for j = 1:Y.ncols
                    Ymat(:,j) = reshape(Y.psi{j},Yn123,1);
                end;
            end;
            [nX,mX]=size(Xmat);
            [nY,mY]=size(Ymat);
            %          if (X.trans==0 & Y.trans==0)
            %             beta = sum(sum(Xmat.*Ymat));
            %          elseif (X.trans==0 & Y.trans==1)
            %             beta = sum(sum(Xmat.*Ymat'));
            %          elseif (X.trans==1 * Y.trans==0)
            %             beta = sum(sum(Xmat'.*Ymat));
            %          else
            %             beta = sum(sum(Xmat'.*Ymat'));
            %          end;
            if (X.trans==0 & Y.trans==0)
                beta = sum(sum(conj(Xmat).*Ymat));
            elseif (X.trans==0 & Y.trans==1)
                beta = sum(sum(conj(Xmat).*Ymat'));
            elseif (X.trans==1 * Y.trans==0)
                beta = sum(sum(conj(Xmat').*Ymat));
            else
                beta = sum(sum(conj(Xmat').*Ymat'));
            end;
        else
            error('The second operand must be a Wavefun object');
        end;
    else
        error('The first operand must be a Wavefun object');
    end;
else
    error('Wavefun: multiplication syntax: Z = X.*Y');
end;

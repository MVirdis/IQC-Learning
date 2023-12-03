function sys = blkdiagss(varargin)
%BLKDIAGSS Block diagonal concatenation of ss systems

A = []; B=[]; C=[]; D=[];

Ts = 0;
for i=1:nargin
    sys_ = varargin{i};
    if i > 1
        assert(Ts == sys_.Ts, 'All systems must have the same sampling-time');
    else
        Ts = sys_.Ts;
    end
    A = blkdiag(A, sys_.A);
    B = blkdiag(B, sys_.B);
    C = blkdiag(C, sys_.C);
    D = blkdiag(D, sys_.D);
end

sys = ss(A,B,C,D,Ts);

end


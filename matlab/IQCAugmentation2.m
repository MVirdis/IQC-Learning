addpath('C:\Program Files\Mosek\9.3\toolbox\R2015a')
addpath(genpath('YALMIP'))

%% Define Standard IQC Analysis Systems

% Here only robust stability is analyzed

T = connect(P,Klqr,{'p1','p2','p3'},{'q1','q2','q3'});

nq1 = 2;
np1 = 2;
nq2 = 1;
np2 = 1;

% Add dynamics to multiplier
nu1 = 2;
rho1 = -1;
psi1 = tf(1)*zeros((nu1+1)*nq1,nq1);
psi1(1:nq1,1:nq1) = tf(1)*eye(nq1);
poly_ = [1,-rho1];
poly = poly_;
for iTF=1:nu1
    psi1((iTF)*nq1+1:(iTF+1)*nq1,1:nq1) = tf(1,poly)*eye(nq1);
    poly = conv(poly,poly_);
end
Anu1 = ss(psi1).A;
Bnu1 = ss(psi1).B;
Cnu1 = ss(psi1).C;
Dnu1 = ss(psi1).D;

nu2 = 2;
rho2 = -1;
psi2 = tf(1)*zeros((nu2+1)*nq2,nq2);
psi2(1:nq2,1:nq2) = tf(1)*eye(nq2);
poly_ = [1,-rho2];
poly = poly_;
for iTF=1:nu2
    psi2((iTF)*nq2+1:(iTF+1)*nq2,1:nq2) = tf(1,poly)*eye(nq2);
    poly = conv(poly,poly_);
end
Anu2 = ss(psi2).A;
Bnu2 = ss(psi2).B;
Cnu2 = ss(psi2).C;
Dnu2 = ss(psi2).D;

psi = blkdiagss(ss(0.2*psi1),ss(0.1*psi2),ss(psi1),ss(psi2));

% Realization of [G_qp;I]
G = [T;ss(eye(np1+np2))];
G = series(G,psi);
A = G.A;
B = G.B;
C = G.C;
D = G.D;

%% Standard IQC

yalmip('clear')

X = sdpvar(length(A),length(A),'symmetric','real');
P11_1 = sdpvar(nq1*(nu1+1),nq1*(nu1+1),'symmetric','real');
P11_2 = sdpvar(nq2*(nu2+1),nq2*(nu2+1),'symmetric','real');
P12_1 = sdpvar(nq1*(nu1+1),nq1*(nu1+1),'full','real');
P12_2 = sdpvar(nq2*(nu2+1),nq2*(nu2+1),'full','real');
Xnu_1 = sdpvar(nq1*nu1*(nu1+1)/2,nq1*nu1*(nu1+1)/2,'symmetric','real');
Xnu_2 = sdpvar(nq2*nu2*(nu2+1)/2,nq2*nu2*(nu2+1)/2,'symmetric','real');

term1 = [eye(length(A)), zeros(length(A),size(B,2));
         A,  B;
         C,  D];

term2 = blkdiag([zeros(length(A)),X;
                 X,zeros(length(A))], ...
                [blkdiag(P11_1,P11_2), blkdiag(P12_1,P12_2);
                 blkdiag(P12_1,P12_2)', -blkdiag(P11_1,P11_2)]);

term3_1 = [eye(length(Anu1)),zeros(length(Anu1),size(Bnu1,2));
           Anu1, Bnu1;
           Cnu1, Dnu1];
term4_1 = blkdiag([zeros(length(Anu1)),Xnu_1;
                   Xnu_1,zeros(length(Anu1))], ...
                  P11_1);

term3_2 = [eye(length(Anu2)),zeros(length(Anu2),size(Bnu2,2));
           Anu2, Bnu2;
           Cnu2, Dnu2];
term4_2 = blkdiag([zeros(length(Anu2)),Xnu_2;
                   Xnu_2,zeros(length(Anu2))], ...
                  P11_2);

% cnstr1 = P11 >= 0;
cnstr2_1 = (P12_1 == -P12_1');
cnstr2_2 = (P12_2 == -P12_2');
cnstr5 = term1'*term2*term1 <= -1e-4*eye(size(term1,2));
cnstr6 = term3_1'*term4_1*term3_1 >= 1e-4*eye(size(term3_1,2));
cnstr7 = term3_2'*term4_2*term3_2 >= 1e-4*eye(size(term3_2,2));

options = sdpsettings('verbose',1,'solver','mosek');
optimize([cnstr2_1,cnstr2_2,cnstr5,cnstr6,cnstr7],[],options)

%% Define Augmented IQC Analysis Systems

T = connect(P,Klqr,{'p1','p2','p3','u'}, ...
            {'q1','q2','q3','y1','y2','y3','y4'});

nq1 = 2;
np1 = 2;

nq2 = 1;
np2 = 1;

nq3 = 4;
np3 = 1;

% Add dynamics to multiplier
nu1 = 3;
rho1 = -10;
psi1 = tf(1)*zeros((nu1+1)*nq1,nq1);
psi1(1:nq1,1:nq1) = tf(1)*eye(nq1);
poly_ = [1,-rho1];
poly = poly_;
for iTF=1:nu1
    psi1((iTF)*nq1+1:(iTF+1)*nq1,1:nq1) = tf(1,poly)*eye(nq1);
    poly = conv(poly,poly_);
end
Anu1 = ss(psi1).A;
Bnu1 = ss(psi1).B;
Cnu1 = ss(psi1).C;
Dnu1 = ss(psi1).D;

nu2 = 3;
rho2 = -10;
psi2 = tf(1)*zeros((nu2+1)*nq2,nq2);
psi2(1:nq2,1:nq2) = tf(1)*eye(nq2);
poly_ = [1,-rho2];
poly = poly_;
for iTF=1:nu2
    psi2((iTF)*nq2+1:(iTF+1)*nq2,1:nq2) = tf(1,poly)*eye(nq2);
    poly = conv(poly,poly_);
end
Anu2 = ss(psi2).A;
Bnu2 = ss(psi2).B;
Cnu2 = ss(psi2).C;
Dnu2 = ss(psi2).D;

psi = blkdiagss(ss(0.2*psi1),ss(0.1*psi2),ss(eye(nq3)), ...
                ss(psi1),ss(psi2),ss(eye(np3)));

% Realization of [G_qp;I]
G = [T;ss(eye(np1+np2+np3))];
G = series(G,psi);
A = G.A;
B = G.B;
C = G.C;
D = G.D;

%% Solve Augmented IQC Analysis Problem

yalmip('clear')

X = sdpvar(length(A),length(A),'symmetric','real');
P11_1 = sdpvar(nq1*(nu1+1),nq1*(nu1+1),'symmetric','real');
P11_2 = sdpvar(nq2*(nu2+1),nq2*(nu2+1),'symmetric','real');
P12_1 = sdpvar(nq1*(nu1+1),nq1*(nu1+1),'full','real');
P12_2 = sdpvar(nq2*(nu2+1),nq2*(nu2+1),'full','real');
Xnu_1 = sdpvar(nq1*nu1*(nu1+1)/2,nq1*nu1*(nu1+1)/2,'symmetric','real');
Xnu_2 = sdpvar(nq2*nu2*(nu2+1)/2,nq2*nu2*(nu2+1)/2,'symmetric','real');
R = sdpvar(nq3,nq3,'symmetric','real');
Q = sdpvar(np3,np3,'symmetric','real');
S = sdpvar(nq3,np3,'full','real');

term1 = [eye(length(A)), zeros(length(A),size(B,2));
         A,  B;
         C,  D];

term2 = blkdiag([zeros(length(A)),X;
                 X,zeros(length(A))], ...
                [blkdiag(P11_1,P11_2,R), blkdiag(P12_1,P12_2,S);
                 blkdiag(P12_1',P12_2',S'), blkdiag(-P11_1,-P11_2,Q)]);

term3_1 = [eye(length(Anu1)),zeros(length(Anu1),size(Bnu1,2));
           Anu1, Bnu1;
           Cnu1, Dnu1];
term4_1 = blkdiag([zeros(length(Anu1)),Xnu_1;
                   Xnu_1,zeros(length(Anu1))], ...
                  P11_1);

term3_2 = [eye(length(Anu2)),zeros(length(Anu2),size(Bnu2,2));
           Anu2, Bnu2;
           Cnu2, Dnu2];
term4_2 = blkdiag([zeros(length(Anu2)),Xnu_2;
                   Xnu_2,zeros(length(Anu2))], ...
                  P11_2);

cnstr2_1 = (P12_1 == -P12_1');
cnstr2_2 = (P12_2 == -P12_2');
cnstr5 = term1'*term2*term1 <= -1e-4*eye(size(term1,2));
cnstr6 = term3_1'*term4_1*term3_1 >= 1e-4*eye(size(term3_1,2));
cnstr7 = term3_2'*term4_2*term3_2 >= 1e-4*eye(size(term3_2,2));
cnstr8 = Q <= 0;
cnstr9 = R >= 0;

cost = [R(:);Q(:)]'*eye(17)*[R(:);Q(:)];

options = sdpsettings('verbose',1,'solver','mosek');
optimize([cnstr2_1,cnstr2_2,cnstr5,cnstr6,cnstr7,cnstr8,cnstr9],[],options)

%% Linearized Cart-Pole system with uncertain pole and cart masses

% Params
Mp = 1;
Mc = 2;
g = 9.81;
l = 1;

% State-space matrices in LFT form
A = [0, 1, 0, 0;
     0, 0, -Mp/Mc*g, 0;
     0, 0, 0, 1;
     0, 0, g/l+(Mp*g)/(Mc*l), 0];

B1 = [0,0,0;
      -1,0,-1/Mc;
      0,0,0;
      0,-1, 1/(Mc*l)];

B2 = [0;1/Mc;0;-1/Mc];

C1 = [0, 0, -(Mp*g)/(Mc), 0;
      0, 0, (Mp*g)/(Mc*l), 0;
      0, 0, Mp*g, 0];

D11 = [-1, 0, -1/Mc;
       0,-1,1/(Mc*l);
       0,0,0];

D12 = [1/Mc;-1/Mc;Mp*g];

C2 = eye(4);

D21 = [0,0,0; 0,0,0; 0,0,0; 0,0,0];

D22 = [0;0;0;0];

C3 = zeros(1,4);

D31 = zeros(1,3);

D32 = 1;

C4 = eye(4);

D41 = [0,0,0;0,0,0;0,0,0;0,0,0];

D42 = [0;0;0;0];

P = ss(A, [B1,B2], [C1;C2;C3;C4], [D11, D12; D21, D22; D31, D32; D41,D42]);

P.InputName = {'p1','p2','p3','u'};
P.OutputName = {'q1','q2','q3','x1','x2','x3','x4','u_',...
                'y1','y2','y3','y4'};
P

%% Synthesis of LTI controller for LQR cost (Nominal and Robust)

robust_flag = 1;

dMc = ureal('dMc',0,'Range',[-0.2,0.2]);
dMp = ureal('dMp',0,'Range',[-0.1,0.1]);

Punc = lft([dMc*eye(2),zeros(2,1);zeros(1,2),dMp],P,3,3);
% Punc.InputName = {'u','w1','w2','w3','w4','v1','v2','v3','v4'};
% Punc.OutputName = {'y1','y2','y3','y4','u_'};

Pnom = Punc.NominalValue;

% Define LQR cost
Q = diag([10, 0.1, 10, 0.1]);
R = 0.01;

if ~robust_flag
    [Klqr, Sare, clps] = lqr(Pnom(1:4,1),Q,R);
    Klqr = ss(-Klqr); % Negative feedback
    Klqr.InputName = {'y1','y2','y3','y4'};
    Klqr.OutputName = {'u'};
else
    % Robust LQR controller
    Klqr = tunableSS('Klqr',4,1,4);
    Klqr.OutputName = {'u'};
    Klqr.InputName = {'y1','y2','y3','y4'};
    
    LQG_req = TuningGoal.LQG({'u'}, {'x1','x2','x3','x4'}, ...
                             0.01*eye(4), diag([10,0.1,10,0.1]));
    Ptune = connect(Punc,Klqr,{'u'}, {'y1','y2','y3','y4'});
    
    Options = systuneOptions('RandomStart',40);
    rng(0)
    [CL,~,~] = systune(Ptune,[],[],Options);
    Klqr = getBlockValue(CL,'Klqr');
    Klqr.OutputName = {'u'};
    Klqr.InputName = {'y1','y2','y3','y4'};
end

%% Make datasets
M = connect(Punc,Klqr,{'u'},{'x1','x2','x3','x4','u_'});

numSamples = 999;
[Msamples,deltaSamples] = usample(M,numSamples);

% Check instability empirically
for iSys=1:numSamples
    Msample = Msamples(:,:,iSys);
    if any(real(eig(Msample.A)) > 0)
        fprintf('Unstable plant %d detected\n',iSys);
    end
end

M_ = c2d(M.NominalValue,0.1,'zoh');
A_ = reshape(M_.A,[1,size(M_.A)]);
B_ = reshape(M_.B,[1,size(M_.B)]);
C_ = reshape(M_.C,[1,size(M_.C)]);
D_ = reshape(M_.D,[1,size(M_.D)]);
dT = 0.1;
N = 1;
dMc = 0;
dMp = 0;

if robust_flag
    M_ = c2d(Msamples,0.1,'zoh');
    
    A = permute(M_.A, [3,1,2]);
    B = permute(M_.B, [3,1,2]);
    C = permute(M_.C, [3,1,2]);
    D = permute(M_.D, [3,1,2]);

    % First ss is the nominal one
    A = cat(1,A_,A);
    B = cat(1,B_,B);
    C = cat(1,C_,C);
    D = cat(1,D_,D);

    dT = 0.1;
    N = numSamples+1;
    dMc = [0,deltaSamples.dMc]';
    dMp = [0,deltaSamples.dMp]';
    save 'robust' 'A' 'B' 'C' 'D' 'N' 'dT' 'dMc' 'dMp';
else
    A = A_; B = B_; C = C_; D = D_;
    save 'nominal' 'A' 'B' 'C' 'D' 'N' 'dT' 'dMc' 'dMp';
end

%% Evaluate LQG using genss

Ntraj = 50;
trajDur = 1e2;          % [s]
discreteStep = 5e-2;    % [s]

rng(0)

Msamples = usample(M,Ntraj);

Msample = M.NominalValue;
Msample = c2d(Msample,discreteStep,'zoh');

perfs = zeros(Ntraj,1);

H = waitbar(0,'Evaluating LQR cost');

for iSys=1:Ntraj
    
    waitbar(iSys/Ntraj,H,sprintf('Evaluating LQR cost (%d%%)',iSys/Ntraj*100));

%     % Discretize
%     Msample = Msamples(:,:,iSys);
%     Msample = c2d(Msample,discreteStep,'zoh');
    
    % Rollout of system
    ts = 0:discreteStep:trajDur;
%     x0 = zeros(length(Msample.A),1);
%     x_ = x0;
%     ys = zeros(length(ts),4);
%     us = zeros(length(ts),1);
%     cost_t = 0;
%     ws = randn(4,length(ts));
%     vs = randn(4,length(ts));
%     for it=1:length(ts)
%         out = Msample.C*x_+Msample.D*[ws(:,it);vs(:,it)];
%         ys(it,:) = out(1:4,1)';
%         us(it,:) = out(5,1)';
%         cost_t = cost_t + out(1:4,1)'*Q*out(1:4,1) + out(5,1)'*R*out(5,1);
%         x_ = Msample.A*x_ + Msample.B*[ws(:,it);vs(:,it)];
%     end
    us = zeros(length(ts),1);
    % Random initial conditions
    x0 = zeros(length(Msample.A),1);
    x0(1) = rand()*(10+10) -10 ;       % in [-10,10]
    x0(2) = rand()*(0.5+0.5) -0.5 ;    % in [-0.5,0.5]
    x0(3) = rand()*(2+2) -2 ;          % in [-2,2]
    x0(4) = rand()*(0.5+0.5) -0.5 ;    % in [-0.5,0.5]
    outs = lsim(Msample, us, [], x0);
    ys = outs(:,1:4);
    us = outs(:,5);

    if any(any(isnan(outs)))
        fprintf('%d\n',iSys);
    end
    
    % Compute empirical Q-R cost over trajectory
%     perfs(iSys) = 1/(length(ts))*(trace(ys*Q*ys') + trace(us*R*us'));
    cost_t = 0;
    for it=1:length(ts)
        cost_t = cost_t + (ys(it,:)*Q*ys(it,:)' + us(it,:)*R*us(it,:)'-cost_t)/it;
    end
    perfs(iSys) = cost_t;
end

mean(perfs)
% hist(perfs)

return
%%

N = connect(Punc,Klqr,{'u'},{'x1','x2','x3','x4','u_'}).NominalValue;
N = c2d(N,0.1,'zoh');
A = N.A; B = N.B; C = N.C;
save 'Sys_matrices' 'A' 'B' 'C';

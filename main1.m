close all; clear all; clc;

% 18 Jan 2017
% Trying out PCA on some simple data
ftol = 99; % percentile for monitoring statistics

% Generate some random input data vectors:
n = 5000; % number of samples
t = 1:n; % time stamp vector
t = t(:); % column vector

% specify mu for each input:
mA = 2;
mB = 1;
mC = 5;

% specify sigma for each input:
sA = 0.05;
sB = 0.09;
sC = 0.03;

% generate inputs:
A = zeros(size(t))+mA;
B = zeros(size(t))+mB;
C = zeros(size(t))+mC;

% Use random walks to generate inputs:
for i = 2:n
    A(i) = A(i-1) + sA*(rand(1,1)-0.5);
    B(i) = B(i-1) + sB*(rand(1,1)-0.5);
    C(i) = C(i-1) + sC*(rand(1,1)-0.5);
end

% % Generate two sets of normal operating data, with differing ICs:
% for i = 2:n-floor(n/2)
%     A(i) = A(i-1) + sA*(rand(1,1)-0.5);
%     B(i) = B(i-1) + sB*(rand(1,1)-0.5);
%     C(i) = C(i-1) + sC*(rand(1,1)-0.5);
% end
% 
% A(i) = A(i)*1.5;
% 
% for i = n-floor(n/2-1):n
%     A(i) = A(i-1) + sA*(rand(1,1)-0.5);
%     B(i) = B(i-1) + sB*(rand(1,1)-0.5);
%     C(i) = C(i-1) + sC*(rand(1,1)-0.5);
% end

% generate linear combinations D, E and F from A,B,C:
mD = 0.3;
cD = 1;
D = mD*A + cD;
D = D + D.*(rand(size(D)) - 0.5)*2/100; % introduce 2% error into this measurement

mEA = -2;
mEB = 1.4;
cE = 0;
E = mEA*A + mEB*B + cE;
E = E + E.*(rand(size(E)) - 0.5)*2/100; % introduce 2% error into this measurement

mFD = 3.5;
mFE = 0.9;
mFC = -0.2;
cF = 1.3;
F = mFD*D + mFE*E + mFC*C + cF;
F = F + F.*(rand(size(F)) - 0.5)*2/100; % introduce 2% error into this measurement

% extract mu and sigma from data:
mAd = mean(A);
mBd = mean(B);
mCd = mean(C);
mDd = mean(D);
mEd = mean(E);
mFd = mean(F);
sAd = std(A);
sBd = std(B);
sCd = std(C);
sDd = std(D);
sEd = std(E);
sFd = std(F);

% shift all data to zero mean and unit variance:
Ad = (A - mAd)/sAd;
Bd = (B - mBd)/sBd;
Cd = (C - mCd)/sCd;
Dd = (D - mDd)/sDd;
Ed = (E - mEd)/sEd;
Fd = (F - mFd)/sFd;

% Data set is inputs and only final output (D,E are invisible)
myD = [Ad Bd Cd Fd]; % centred data

myD = [Ad Bd Cd Dd Ed Fd];

% determine PCs, scores, eigenvalues, T^2 of data set:
[Ppc,Spc,lt] = pca(myD);

% extract % explained by each PC:
exp = lt/sum(lt)*100;
NC = find(exp<5,1)-1; % retain # of PCs to explain >95% of variation

% Define PC and residual spaces:
P = Ppc(:,1:NC);
Pr = Ppc(:,NC+1:end);
sc = Spc(:,1:NC);
scr = Spc(:,NC+1:end);

% reconstruct data
rec = sc*P'; % reconstruction in PC space
recr = scr*Pr'; % in residual space

idx = kmeans(rec,3,'Replicates',10);

% figure;
% plot(t(idx==1),rec(idx==1,6),'.',t(idx==2),rec(idx==2,6),'x',t(idx==3),rec(idx==3,6),'o')

figure;
plot(sc(idx==1,1),sc(idx==1,2),'.',sc(idx==2,1),sc(idx==2,2),'.',sc(idx==3,1),sc(idx==3,2),'.')
xlabel('PC 1')
ylabel('PC 2')

figure;
plot3(sc(idx==1,1),sc(idx==1,2),sc(idx==1,3),'.',sc(idx==2,1),sc(idx==2,2),sc(idx==2,3),'.',sc(idx==3,1),sc(idx==3,2),sc(idx==3,3),'.')
% plot3(sc(idx==1,1),sc(idx==1,2),sc(idx==1,3),'.',sc(idx==2,1),sc(idx==2,2),sc(idx==2,3),'.')

return
% determine SPE:
SPE = sum(recr.^2,2);
SPEb = prctile(SPE,ftol);
mSPE = SPE>SPEb;

% figure;
% plot(t(~mSPE),SPE(~mSPE),'.',t(mSPE),SPE(mSPE),'x')

% % use analytical definition of SPE limit bSPE (ALcala & Qin 2009):
% % bSPE = gSPE*CHI_alpha^2(hSPE). Confidence level (1-alpha)*100%
% % gSPE = sigma2/sigma1
% % hSPE = sigma1^2/sigma2
% % sigma1 = sum_i=l+1:n lambda_i
% % sigma2 = sum_i=l+1:n lambda_i^2
% % l is the number of retained PCs
% % lambda_i is ith eigenvalue
% 
% sig1 = sum(lt(NC+1:end));
% sig2 = sum(lt(NC+1:end).^2);
% hSPE = sig1^2/sig2;
% gSPE = sig2/sig1;
% bSPE = gSPE*chi2pdf(hSPE,100-ftol);
% mbSPE = SPE>bSPE; % ie faulty conditions
% figure;
% plot(t(~mbSPE),SPE(~mbSPE),'.',t(mbSPE),SPE(mbSPE),'x')

% determine Hotelling's T^2:
T2 = mahal(sc,sc); % using help pca's suggested method
T2b = prctile(T2,ftol);
mT = T2>T2b;

% combined threshold statistic:
Cs = (SPE/SPEb + T2/T2b);
Csb = prctile(Cs,ftol);
mCsb = Cs>Csb;

% figure;
% plot(t,Cs,'.',t(mSPE),Cs(mSPE),'xr',t(mT),Cs(mT),'og',t(mCsb),Cs(mCsb),'>b')
% break

% % This code generates T^2 more manually and slightly slower than mahal:
% lt = lt(1:NumComp); % retain only eigenvalues for number of PCs
% sc2 = sc.^2; % squared scores
% T2ma = zeros(size(sc(:,1)));
% for i = 1:length(T2ma)
%     T2ma(i) = sum(sc2(i,:)./lt');
% end

%=========================================================================
% Generate some test data:
t2 = n+1:n+500;
t2 = t2(:);

% generate inputs:
% At = zeros(size(t2))+A(end);
% Bt = zeros(size(t2))+B(end);
% Ct = zeros(size(t2))+C(end);
At = zeros(size(t2))+mA;
Bt = zeros(size(t2))+mB;
Ct = zeros(size(t2))+mC;

% Use random walks to generate inputs:
for i = 2:length(t2)
    At(i) = At(i-1) + sA*(rand(1,1)-0.5);
    Bt(i) = Bt(i-1) + sB*(rand(1,1)-0.5);
    Ct(i) = Ct(i-1) + sC*(rand(1,1)-0.5);
end

% generate linear combinations D, E and F from A,B,C:
mD = 0.3;
cD = 1;
Dt = mD*At + cD;
Dt = Dt + Dt.*(rand(size(Dt)) - 0.5)*2/100; % introduce 2% error into this measurement

mEA = -2;
mEB = 1.4;
cE = 0;
Et = mEA*At + mEB*Bt + cE;
Et = Et + Et.*(rand(size(Et)) - 0.5)*2/100; % introduce 2% error into this measurement

mFD = 3.5;
mFE = 0.9;
mFC = -0.2;
cF = 1.3;
Ft = mFD*Dt + mFE*Et + mFC*Ct + cF;
Ft = Ft + Ft.*(rand(size(Ft)) - 0.5)*2/100; % introduce 2% error into this measurement

% shift all data to zero mean and unit variance (using original data mu and sigma):
Adt = (At - mAd)/sAd;
Bdt = (Bt - mBd)/sBd;
Cdt = (Ct - mCd)/sCd;
Ddt = (Dt - mDd)/sDd;
Edt = (Et - mEd)/sEd;
Fdt = (Ft - mFd)/sFd;

myDt = [Adt Bdt Cdt Fdt];
myDt = [Adt Bdt Cdt Ddt Edt Fdt];

% map test data onto principal components:
sct = myDt*P;

% reconstruct test data:
rect = sct*P';

% determine SPE:
SPEt = sum((rect - myDt).^2,2);
mt = SPEt>SPEb;

% determine T^2:
% % This code generates T^2 more manually and slightly slower than mahal,
% % but has to be used for evaluating data:
ltd = lt(1:NC); % retain only eigenvalues for number of PCs

sc2 = sct.^2; % squared scores
T2t = zeros(size(sct(:,1)));
for i = 1:length(T2t)
    T2t(i) = sum(sc2(i,:)./ltd');
end

mTt = T2t>T2b;

%=========================================================================
% Generate some error data:
te = t2(end)+1:t2(end)+500;
te = te(:);

% generate inputs:
% Ae = zeros(size(te))+At(end);
% Be = zeros(size(te))+Bt(end);
% Ce = zeros(size(te))+Ct(end);
Ae = zeros(size(te))+mA;
Be = zeros(size(te))+mB;
Ce = zeros(size(te))+mC; % change an input

% Use random walks to generate inputs:
for i = 2:length(te)
    Ae(i) = Ae(i-1) + sA*(rand(1,1)-0.5);
    Be(i) = Be(i-1) + sB*(rand(1,1)-0.5);
    Ce(i) = Ce(i-1) + sC*(rand(1,1)-0.5);
end

% generate linear combinations D, E and F from A,B,C:
mDe = mD*1.2; % fundamental change in process
cD = 1;
De = mDe*Ae + cD;
De = De + De.*(rand(size(De)) - 0.5)*2/100; % introduce 2% error into this measurement

mEA = -2;
mEB = 1.4;
cE = 0;
Ee = mEA*Ae + mEB*Be + cE;
Ee = Ee + Ee.*(rand(size(Ee)) - 0.5)*2/100; % introduce 2% error into this measurement

mFD = 3.5;
mFE = 0.9;
mFC = -0.2;
cF = 1.3;
Fe = mFD*De + mFE*Ee + mFC*Ce + cF;
Fe = Fe + Fe.*(rand(size(Fe)) - 0.5)*2/100; % introduce 2% error into this measurement

% Ae = Ae*0.5; % measurement error on input
% Fe = Fe*0.5; % measurement error on output

% shift all data to zero mean and unit variance (using original data mu and sigma):
Ade = (Ae - mAd)/sAd;
Bde = (Be - mBd)/sBd;
Cde = (Ce - mCd)/sCd;
Dde = (De - mDd)/sDd;
Ede = (Ee - mEd)/sEd;
Fde = (Fe - mFd)/sFd;

myDe = [Ade Bde Cde Fde];
myDe = [Ade Bde Cde Dde Ede Fde];

% map error data onto principal components:
sce = myDe*P; % in PC space
scre = myDe*Pr; % in residual space

% reconstruct error data:
rece = sce*P'; % in PC space
recre = scre*Pr'; % in residual space

% determine SPE:
SPEe = sum((rece - myDe).^2,2);
me = SPEe>SPEb;

% determine T^2:
% % This code generates T^2 more manually and slightly slower than mahal,
% % but has to be used for evaluating data:
sc2 = sce.^2; % squared scores
T2e = zeros(size(sce(:,1)));
for i = 1:length(T2e)
    T2e(i) = sum(sc2(i,:)./ltd');
end
mTe = T2e>T2b;

%=========================================================================
% Determine contributions for error data:

% contributions to SPE:
Cdash = Pr*Pr'; % projection matrix to residual space
cii = diag(Cdash)'; % diagonal elements of Cdash
cont = zeros(size(rece));
RBC = cont;
for i = 1:size(recre,2)
    cont(:,i) = recre(:,i).^2; % standard contribution
    RBC(:,i) = cont(:,i)/cii(i); % Reconstruction-based contribution to SPE
end

figure;
plot(1:length(lt),mean(cont(me,:))/sum(mean(cont(me,:))),1:length(lt),mean(RBC(me,:))/sum(mean(RBC(me,:))))
legend('Contribution','RBC')

% figure;
% subplot(2,1,1)
% plot(t,myD(:,1:3))
% ylabel('Inputs A, B, C')
% legend('A','B','C')
% subplot(2,1,2)
% plot(t,myD(:,4:6))
% ylabel('Output F')
% break

%=========================================================================
% Plot some results:

figure;
subplot(2,1,1)
plot(t,myD(:,1:3),t2,myDt(:,1:3),'.',te,myDe(:,1:3),'x')
ylabel('Inputs A, B, C')
legend('A','B','C')
subplot(2,1,2)
plot(t,myD(:,4),t2,myDt(:,4),'.',te,myDe(:,4),'x')
ylabel('Output F')

figure;
subplot(2,1,1)
plot(t(mSPE),SPE(mSPE),'xr',t(~mSPE),SPE(~mSPE),'.b',...    Training data
    t2(mt),SPEt(mt),'xr',t2(~mt),SPEt(~mt),'.g',...         Test data
    te(me),SPEe(me),'xr',te(~me),SPEe(~me),'.c', ...        Error data
    [min([t; te; t2]) max([t; te; t2])],[SPEb SPEb],'--k')
xlabel('Time')
ylabel('Squared Prediction Error')
subplot(2,1,2)
plot(t(mT),T2(mT),'xr',t(~mT),T2(~mT),'.b',...              Training data
    t2(mTt),T2t(mTt),'xr',t2(~mTt),T2t(~mTt),'.g',...       Test data
    te(mTe),T2e(mTe),'xr',te(~mTe),T2e(~mTe),'.c',...       Error data
    [min([t; te; t2]) max([t; te; t2])],[T2b T2b],'--k')
xlabel('Time')
ylabel('Hotelling''s T^2 error')

% figure;
% plot(t,(SPE/SPEb + T2/T2b),t2,(SPEt/SPEb + T2t/T2b),te,(SPEe/SPEb + T2e/T2b),[0 max([t;t2;te])],[Csb Csb],'b--')
% 
% figure;
% plot(sc(:,1),sc(:,2),'.b',sct(:,1),sct(:,2),'xc',sce(:,1),sce(:,2),'xr')
% legend('Training','Validation','Error')
% 
% figure;
% plot3(sc(:,1),sc(:,2),sc(:,3),'.b')
% hold on
% plot3(sct(:,1),sct(:,2),sct(:,3),'.c')
% hold on
% plot3(sce(:,1),sce(:,2),sce(:,3),'xr')
% xlabel('PC1')
% ylabel('PC2')
% zlabel('PC3')
% This code takes in input a matrix v(mn) containing the number of
% visualizations v(m,n) that a video m obtains after n hours. It gives as
% output linear and multiple linear regression models predicting the number
% of visualisations of a video after 168 hours by using the data relative
% to n from 1 to 24.

clear all
close all

% some constant definitions
sgl = [0.682, 0.954, 0.997] ;   % 1-2-3 sigma values


v = csvread('data.csv',0,1) ;   % reading the data matrix starting from 
                                % column number 2
[m, ~] = size(v) ;              % setting the data matrix sizes
                                % m = video number; n = hours


%% basic statistics of the visualization distribution at given times t1 

t1 = [24, 72, 168] ;                    % time of the distributions to analyse
ln_t1 = length(t1) ;                    % t1 vector lenght

figure('Position',[0,0,1200,650])

for aa = 1:ln_t1
    vec  = v(:,t1(aa)) ;                % analysed sample
    sd   = std(vec) ;                   % standard deviation of the sample
    md   = median(vec) ;                % median of the sample  
    mn   = mean(vec) ;                  % mean of the sample
    
    
    [dist, xdist] = hist(vec,1000) ;    % computation of the distribution
    [~, imo]      = max(dist) ;         
    mo            = xdist(imo) ;        % mode of the distribution
    
    subplot(2,3,aa)
    plot(vec,'.','markersize',10)
    xlim([1 m])
    line([1 m],[mn mn],'Color','r')
    line([1 m],[mn+sd mn+sd],'Color','k','linestyle','-.')
    line([1 m],[mn+2*sd mn+2*sd],'Color','k','linestyle',':')
    line([1 m],[mn+3*sd mn+3*sd],'Color','k','linestyle','--')
    legend({'data','mean','1 \sigma','2 \sigma','3 \sigma'},'FontSize',8, 'location','northwest')
    title(['t = ',num2str(t1(aa)),' [h]'],'FontSize',14)
    xlabel('Video ID','FontSize',14)
    ylabel(['v(:,',num2str(t1(aa)),')'],'FontSize',14)
    set(gca,'FontSize',14)
    
    subplot(2,3,aa+3)
    bar(xdist,dist)
    line([mn mn],[0 0.6*max(dist)],'Color','r')
    text(mn,0.6*max(dist),['mean = ',num2str(round(mn))],'color','r')
    line([md md],[0 0.8*max(dist)],'Color','g')
    text(md,0.8*max(dist),['median = ',num2str(round(md))],'color','g')
    line([mo mo],[0 1.0*max(dist)],'Color','k')
    text(mo,1.0*max(dist),['mode = ',num2str(round(mo))],'color','k')
    xlim([xdist(1)*0.7 mn+sd])%xdist(isgl)])
    xlabel(['v(:,',num2str(t1(aa)),')'],'FontSize',14)
    ylabel('Counts','FontSize',14)
    set(gca,'FontSize',14)
end
fileou = strcat('Basic_Statistics.eps') ;
set(gcf,'PaperPositionMode','auto') ;
print('-depsc','-r300',fileou) 

clear vec sd md mn imo mo dist xdist 

%% distribution of the case t = 168

vec  = v(:,168) ;                 % analysed sample
sd   = std(vec)   ;                 % standard deviation of the sample
mn   = mean(vec)  ;                 % mean of the sample
md   = median(vec) ;                % median of the sample  

[dist, xdist] = hist(vec,1000) ;    % distribution computation
[~, imo]      = max(dist) ;         
mo            = xdist(imo) ;        % mode of the distribution

figure('Position',[0,0,1200,650])

subplot(2,2,1)
plot(vec,'.','markersize',10)
xlim([1 m])
line([1 m],[mn mn],'Color','r')
line([1 m],[mn+sd mn+sd],'Color','k','linestyle','-.')
line([1 m],[mn+2*sd mn+2*sd],'Color','k','linestyle',':')
line([1 m],[mn+3*sd mn+3*sd],'Color','k','linestyle','--')
legend({'data','mean','1 \sigma','2 \sigma','3 \sigma'},'FontSize',8, 'location','northwest')
title(['t = ',num2str(168),' [h]'],'FontSize',14)
xlabel('Video ID','FontSize',14)
ylabel(['v(:,',num2str(t1(aa)),')'],'FontSize',14)
set(gca,'FontSize',14)

subplot(2,2,3)
bar(xdist,dist)
line([mn mn],[0 0.6*max(dist)],'Color','r')
text(mn,0.6*max(dist),['mean = ',num2str(round(mn))],'color','r')
line([md md],[0 0.8*max(dist)],'Color','g')
text(md,0.8*max(dist),['median = ',num2str(round(md))],'color','g')
line([mo mo],[0 1.0*max(dist)],'Color','k')
text(mo,1.0*max(dist),['mode = ',num2str(round(mo))],'color','k')
xlim([xdist(1)*0.7 mn+sd])%xdist(isgl)])
xlabel(['v(:,',num2str(t1(aa)),')'],'FontSize',14)
ylabel('Counts','FontSize',14)
set(gca,'FontSize',14)

clear vec sd md mn imo mo dist xdist 


lvec = log10(v(:,168)) ;             % analysed sample
sd   = std(lvec)   ;                 % standard deviation of the sample
mn   = mean(lvec)  ;                 % mean of the sample
md   = median(lvec) ;                % median of the sample  

[ldist, lxdist] = hist(lvec,50) ;    % distribution computation
[~, imo]        = max(ldist) ;         
mo              = lxdist(imo) ;      % mode of the distribution

subplot(2,2,2)
plot(lvec,'.','markersize',10)
xlim([1 m])
line([1 m],[mn mn],'Color','r')
line([1 m],[mn+sd mn+sd],'Color','k','linestyle','-.')
line([1 m],[mn+2*sd mn+2*sd],'Color','k','linestyle',':')
line([1 m],[mn+3*sd mn+3*sd],'Color','k','linestyle','--')
line([1 m],[mn-sd mn-sd],'Color','k','linestyle','-.')
line([1 m],[mn-2*sd mn-2*sd],'Color','k','linestyle',':')
line([1 m],[mn-3*sd mn-3*sd],'Color','k','linestyle','--')
legend({'data','mean','1 \sigma','2 \sigma','3 \sigma'},'FontSize',8, 'location','northwest')
title(['log10 transformed visualization for t = ',num2str(168),' [h]'],'FontSize',14)
xlabel('Video ID','FontSize',14)
ylabel(['log10(v(:,',num2str(t1(aa)),'))'],'FontSize',14)
set(gca,'FontSize',14)

subplot(2,2,4)
bar(lxdist,ldist)
line([mn mn],[0 0.6*max(ldist)],'Color','r')
text(mn,0.6*max(ldist),['mean = ',num2str(mn)],'color','r')
line([md md],[0 0.8*max(ldist)],'Color','g')
text(md,0.8*max(ldist),['median = ',num2str(md)],'color','g')
line([mo mo],[0 1.0*max(ldist)],'Color','k')
text(mo,1.0*max(ldist),['mode = ',num2str(mo)],'color','k')
%xlim([xdist(1)*0.7 mn+sd])%xdist(isgl)])
xlabel(['log10(v(:,',num2str(t1(aa)),'))'],'FontSize',14)
ylabel('Counts','FontSize',14)
set(gca,'FontSize',14)

fileou = strcat('168_dists.eps') ;
set(gcf,'PaperPositionMode','auto') ;
print('-depsc','-r300',fileou)

clear lvec sd md mn imo mo ldist lxdist 

%% Outliers cleaning

lvec  = log10(v(:,168))   ;          % analysed sample
sd    = std(lvec)         ;          % standard deviation of the sample
mn    = mean(lvec)        ;          % mean of the sample 

i_cl = find(lvec < (mn + 3*sd) & lvec > mn - 3*sd) ;

clear sd mn md

lvec_cl = lvec(i_cl) ;
sd_cl = std(lvec_cl)         ;          % standard deviation of the sample
mn_cl = mean(lvec_cl)        ;          % mean of the sample 
md_cl = median(lvec_cl)      ;          % median of the sample  

[ldist_cl, lxdist_cl] = hist(lvec_cl,50) ;    % distribution computation
[~, imo]              = max(ldist_cl) ;         
mo_cl                 = lxdist_cl(imo) ;        % mode of the distribution

figure('Position',[0,0,1200,650])

subplot(2,1,1)
plot(lvec_cl,'.','markersize',10)
xlim([1 m])
line([1 m],[mn_cl mn_cl],'Color','r')
line([1 m],[mn_cl+sd_cl mn_cl+sd_cl],'Color','k','linestyle','-.')
line([1 m],[mn_cl+2*sd_cl mn_cl+2*sd_cl],'Color','k','linestyle',':')
line([1 m],[mn_cl+3*sd_cl mn_cl+3*sd_cl],'Color','k','linestyle','--')
line([1 m],[mn_cl-sd_cl mn_cl-sd_cl],'Color','k','linestyle','-.')
line([1 m],[mn_cl-2*sd_cl mn_cl-2*sd_cl],'Color','k','linestyle',':')
line([1 m],[mn_cl-3*sd_cl mn_cl-3*sd_cl],'Color','k','linestyle','--')
legend({'data','mean','1 \sigma','2 \sigma','3 \sigma'},'FontSize',8, 'location','northwest')
title(['3 \sigma cleaned log10 transformed visualization for t = ',num2str(168),' [h]'],'FontSize',14)
xlabel('Video ID','FontSize',14)
ylabel(['log10(v(:,',num2str(t1(aa)),'))'],'FontSize',14)
set(gca,'FontSize',14)

subplot(2,1,2)
bar(lxdist_cl,ldist_cl)
line([mn_cl mn_cl],[0 0.6*max(ldist_cl)],'Color','r')
text(mn_cl,0.6*max(ldist_cl),['mean = ',num2str(mn_cl)],'color','r')
line([md_cl md_cl],[0 0.8*max(ldist_cl)],'Color','g')
text(md_cl,0.8*max(ldist_cl),['median = ',num2str(md_cl)],'color','g')
line([mo_cl mo_cl],[0 1.0*max(ldist_cl)],'Color','k')
text(mo_cl,1.0*max(ldist_cl),['mode = ',num2str(mo_cl)],'color','k')
%xlim([xdist(1)*0.7 mn+sd])%xdist(isgl)])
xlabel(['log10(v(:,',num2str(t1(aa)),'))'],'FontSize',14)
ylabel('Counts','FontSize',14)
set(gca,'FontSize',14)

fileou = strcat('168_dists_3s_clean.eps') ;
set(gcf,'PaperPositionMode','auto') ;
print('-depsc','-r300',fileou)

clear lvec sd md mn sd_cl md_cl mn_cl imo mo ldist lxdist ldist_cl lxdist_cl

%% Correlation coefficients

t2    = 1:1:24     ;
ln_t2 = length(t2) ;

lvec168_cl = lvec_cl ; 

R = zeros(1,ln_t2) ;
P = zeros(1,ln_t2) ;
RL = zeros(1,ln_t2) ;
RU = zeros(1,ln_t2) ;

for bb = 1:ln_t2
    lvec    = log10(v(i_cl,t2(bb))) ; %
    in0     = not(isinf(lvec))      ; % accounts for 0 views in n=1 sample
    
    [Rmtx, Pmtx, RLmtx, RUmtx]   = corrcoef(lvec(in0),lvec168_cl(in0)) ;
    R(bb)  = Rmtx(1,2)  ;
    P(bb)  = Pmtx(1,2)  ;
    RL(bb) = RLmtx(1,2) ;
    RU(bb) = RUmtx(1,2) ;
end

figure('Position',[0,0,900,550])
plot(t2,R,'b','linewidth',2)
hold on
for cc = 1:ln_t2
    line([cc cc],[RL(cc) RU(cc)])
end
xlabel('Time [h]','FontSize',14)
ylabel('R','FontSize',14)
set(gca,'FontSize',14)
title('Pearson correlation coefficien computed between log10(v(:,168)) and log10(v(:,n) for n=1,...,24','FontSize',14)


fileou = strcat('R_1to24_168.eps') ;
set(gcf,'PaperPositionMode','auto') ;
print('-depsc','-r300',fileou)

clear lvec in0 R P RL RU

%% split the whole sample in test and training samples

% cutting the data set for the 3sigma level of the 168 h sample and setting
% to 1 all visualizations equal to 0 (inf in log10) of the 1 h sample
lv_cl    = log10(v(i_cl,:))             ;

sz_vec  = length(lv_cl)  ;
sz_test = round(0.1*sz_vec) ;
sz_trai = sz_vec - sz_test  ;

% Here I split the sample in test and training avoiding index repetitions
i_test = zeros(1,sz_test) ;
i_test(1) = randi([1 sz_vec]) ;
for ff = 2:sz_test
    temp = randi([1 sz_vec]) ;
    while intersect(temp, i_test)
        temp = randi([1 sz_vec]) ;
    end
    i_test(ff) = temp ;
end
    
i_trai = setdiff(1:sz_vec,i_test) ;

test_sam = lv_cl(i_test,:) ;
trai_sam = lv_cl(i_trai,:) ;

%% linear regression 

n = 24 ;
vec_168 = trai_sam(:,168) ;
vec_n   = trai_sam(:,n)   ;

[p,S] = polyfit(vec_n,vec_168,1) ;
angular   = p(1) ;
intercept = p(2) ;

xer = linspace(min(vec_n),max(vec_n),10) ;

[yer,delta] = polyval(p,xer,S) ;

figure('Position',[0,0,900,550])
plot(vec_n,vec_168,'.')
hold on
errorbar(xer,yer,delta,'r','linewidth',1.5)
xlabel(['log10(v(:,',num2str(n),'))'],'FontSize',14)
ylabel(['log10(v(:,',num2str(168),'))'],'FontSize',14)
text(min(vec_n),max(vec_168),['v(:,168) = a * v(:,n) + b; a= ',num2str(p(1)),'; b= ',num2str(p(2))],'FontSize',14)
set(gca,'FontSize',14)
title(['Correlation between v(:,',num2str(n),') and v(:,',num2str(168),') distributions'],'FontSize',14)

fileou = strcat('Lin_reg_168_n.eps') ;
set(gcf,'PaperPositionMode','auto') ;
print('-depsc','-r300',fileou)

%% Multiple linear regression

clear model

n = 24 ;
X = ones(size(trai_sam(:,168))) ;
for tt=1:n
    X = [X trai_sam(:,tt)];
end

b = regress(trai_sam(:,168),X) ;

MLRmodel = b(1) ;
for tt=2:n+1
    MLRmodel = MLRmodel + b(tt)*trai_sam(:,tt-1) ;
end

% PLOT EXAMPLE ###########################################################
Xpl = ones(size(trai_sam(:,168))) ;

Xpl = [Xpl trai_sam(:,23) trai_sam(:,24)];

bpl = regress(trai_sam(:,168),Xpl) ;

plt_vec1 = linspace(min(trai_sam(:,23)),max(trai_sam(:,23)),10) ;
plt_vec2 = linspace(min(trai_sam(:,24)),max(trai_sam(:,24)),10) ;

modelpl = bpl(1) + bpl(2)*plt_vec1 + bpl(3)*plt_vec2  ;

figure('Position',[0,0,900,550])
plot3(trai_sam(:,23),trai_sam(:,24),trai_sam(:,168),'.')
hold on
plot3(plt_vec1,plt_vec2,modelpl,'r','linewidth',1.5)
xlabel('log10(v(:,23))','FontSize',14)
ylabel('log10(v(:,24))','FontSize',14)
zlabel('log10(v(:,168))','FontSize',14)
set(gca,'FontSize',14)
view([-90 50 90])
fileou = strcat('M_Lin_reg_168_n.eps') ;
set(gcf,'PaperPositionMode','auto') ;
print('-depsc','-r300',fileou)

%% mRSE computation
clear MLRmRSE LRmRSE

trse = 24 ;
[T, ~] = size(test_sam) ;

% Linear regression
LRmRSE = zeros(1,trse) ;
for jj = 1:24
    
    n = jj ;
    vec_168 = trai_sam(:,168) ;
    vec_n   = trai_sam(:,n)   ;
    [p,S] = polyfit(vec_n,vec_168,1) ;
    angular   = p(1) ;
    intercept = p(2) ;
    
    
    v_168_est = 10.^(angular * test_sam(:,jj) + intercept) ;
    v_168_rea = 10.^(test_sam(:,168)) ;
    LRmRSE(jj) = sum( ((v_168_est./v_168_rea) -1).^2) / T ;
end



% Multiple Linear regression

n_ex = 1  ; % excluding data of the first n_ex hours
ln_p = 24 ; % total number of hours to compute regressions

test_sam_ct = test_sam(:,1+n_ex:end) ; % Creation of a new data matrix without the
                                       % the view counts after 1 h containing 0s
new_ln = length(test_sam_ct) ;
MLRmRSE = zeros(1,trse-1) ;
for jj = 1:ln_p-n_ex
    X = ones(size(test_sam_ct(:,new_ln))) ;
    for tt=1:jj
        X = [X test_sam_ct(:,tt)];
    end
    
    b = regress(test_sam_ct(:,new_ln),X) ;

    aux = b(1) ;
    for yy=2:jj+1
        aux = aux + b(yy)*test_sam_ct(:,yy-1) ;
    end    
    
    v_168_est = 10.^aux ;
    
    v_168_rea = 10.^(test_sam_ct(:,new_ln)) ;
    MLRmRSE(jj) = sum( ((v_168_est./v_168_rea) -1).^2) / T ;
end

figure('Position',[0,0,900,550])
plot(1:ln_p,LRmRSE(1:ln_p),'linewidth',2)
hold on
plot(n_ex+1:ln_p,MLRmRSE,'--r','linewidth',2)
grid on
legend('Linear regression','Multiple linear regression')
set(gca,'FontSize',14)
xlabel('Time [h]','FontSize',14)
ylabel('mRSE','FontSize',14)
fileou = strcat('mRSE.eps') ;
set(gcf,'PaperPositionMode','auto') ;
print('-depsc','-r300',fileou)
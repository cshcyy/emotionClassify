clear
clc
load('ECGData.mat')
Data = ECGData.Data;
Label = ECGData.Labels;
a= Data(1,:);
b= Data(1,:);
plot(b);
% 原始信号
fs = 128;
figure
subplot(3,1,1);
Ti = 0:1/128:65536/128 - 1/128;
plot(Ti,a)
title('原信号');
set(gca,'FontSize',12);
grid on
grid minor
set(gca, 'MinorGridAlpha', 0.5);
set(gca, 'GridAlpha', 0.5);
hold on;
plot(Ti,b);

% 连续小波变换
 wavename='db4';
% % totalscal=256;
% % Fc=centfrq(wavename); % 小波的中心频率
% % c=2*Fc*totalscal;
% % scals=c./(1:totalscal);
% % f=scal2frq(scals,wavename,1/fs); % 将尺度转换为频率
% % coefs=cwt(a,scals,wavename); % 求连续小波系数
% [coefs,f] = cwt(a,wavename,fs);
% 
% subplot(2,1,2);
% %imagesc(Ti,f,abs(coefs));
% mesh(Ti,f,abs(coefs));
% %view([0 -90 0])
% set(gca,'YDir','normal')
% colorbar;
% xlabel('时间 t/s');
% ylabel('频率 f/Hz');
% title('小波时频图');
% grid on
% grid minor

%离散小波变换
[cA,cD] = dwt(a,wavename);
subplot(3,1,2);
plot(cA);
subplot(3,1,3);
plot(cD);

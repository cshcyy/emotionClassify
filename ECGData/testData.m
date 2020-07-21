clear
clc
load('ECGData.mat')
Data = ECGData.Data;
Label = ECGData.Labels;
a= Data(1,:);
plot(Data(1,:))
a = dir('C:\Omri\BGU\2017\deep learning\finalProject_2016\finalProject.students\finalProject.students\data\Test\SEG\*.png')
for i=1:480
   disp(i);
   fn = fullfile('C:\Omri\BGU\2017\deep learning\finalProject_2016\finalProject.students\finalProject.students\data\Test\SEG\',a(i).name);
   Iseg = imread(fn);
   if nnz(Iseg)~=0
       break;
   end    
end

%%
Iseg = imread('C:\Omri\BGU\2017\deep learning\finalProject_2016\finalProject.students\finalProject.students\data\Test\SEG\Alon_Lab_H1299_t_10_y_2_x_1.png','png');
[Iraw,map] = imread('C:\Omri\BGU\2017\deep learning\finalProject_2016\finalProject.students\finalProject.students\data\Test\RAW\Alon_Lab_H1299_t_10_y_2_x_1.png','png');

close all
figure; 
subplot(211);
imshow(Iraw, map);
subplot(212);
Iseg(Iseg==1)=255;
imshow(Iseg);

%%
root_dir = '/home/omri/omri_dl1/play_ground/data_aug';
path = {'Train/RAW/Alon_Lab_H1299_t_72_y_8_x_3.png',
'Train/RAW/Alon_Lab_H1299_t_72_y_8_x_3_90.png',
'Train/RAW/Alon_Lab_H1299_t_72_y_8_x_3_180.png',
'Train/RAW/Alon_Lab_H1299_t_72_y_8_x_3_270.png',
'Train/RAW/Alon_Lab_H1299_t_72_y_8_x_3_0_flip.png',
'Train/RAW/Alon_Lab_H1299_t_72_y_8_x_3_90_flip.png',
'Train/RAW/Alon_Lab_H1299_t_72_y_8_x_3_180_flip.png',
'Train/RAW/Alon_Lab_H1299_t_72_y_8_x_3_270_flip.png'};

figure;
for z=1:numel(path)
    [I,map] = imread(fullfile(root_dir,path{z}));
    if z<5
        subplot(2,4,z);
    else
        subplot(2,4,z);
    end
    imshow(I,map);
end


%% view test image results
path = {
'/home/omri/omri_dl1/play_ground/data_aug/Test/RAW/Alon_Lab_H1299_t_1_y_1_x_2.png',
'/home/omri/omri_dl1/play_ground/data_aug/Test/SEG/Alon_Lab_H1299_t_1_y_1_x_2.png',
'/tmp/outfile.jpg',
};

figure;
for z=1:numel(path)
    [I,map] = imread(path{z});
    subplot(3,1,z);
    imshow(I,map);
end

%% view with predict
close all;
path = {
'/home/omri/omri_dl1/play_ground/data_aug/Test/RAW/Alon_Lab_H1299_t_1_y_3_x_3.png'
'/home/omri/omri_dl1/play_ground/data_aug/Test/SEG/Alon_Lab_H1299_t_1_y_3_x_3.png'
'/tmp/omri/pred3.jpg',};
figure;
for z=1:numel(path)
    [I,map] = imread(path{z});
    subplot(3,1,z);
    imshow(I,map);
end

%%
close all;
filename = '/home/omri/omri_dl1/play_ground/data_aug/test.csv';
delimiter = ',';
formatSpec = '%s%s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
fclose(fileID);
RAWn = dataArray{:, 1};
SEGn = dataArray{:, 2};
clearvars filename delimiter formatSpec fileID dataArray ans;

for k=31:40
% k=9;
RESn = sprintf('/tmp/omri/pred%d.jpg',k-1);
path = {
    fullfile('/home/omri/omri_dl1/play_ground/data_aug/',RAWn{k}),
    fullfile('/home/omri/omri_dl1/play_ground/data_aug/',SEGn{k}),
    RESn};

figure;
for z=1:numel(path)
    [I,map] = imread(path{z});
    subplot(3,1,z);
    imshow(I,map);
end

end


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
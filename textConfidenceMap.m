clear all;
close all;

% parameter setting
a = 40;
b = 30;
t1 = 15;
t2 = 40;
t3 = -2;
t4 = 3;

I = imread('ArabicSign-00071-small.jpg');
I = uint8(rgb2gray(I));
Sx = [-1 0 1;-2 0 2;-1 0 1];
Sy = Sx';
dx = imfilter(I,Sx);
dy = imfilter(I,Sy);
[img_row, img_col] = size(I);
I_dns = zeros(img_row, img_col); % edge dense
I_eov = zeros(img_row, img_col); % edge orientation variance
I_oep = zeros(img_row, img_col); % opposite edge pairs
win_row = 2*b+1;                 % it should be odd (height)
win_col = 2*a+1;                 % it should be odd (width)

tic;
% cal orientation map
% I_orn_t (orientation: theta)
% I_orn_l (orientation: lumda)
I_orn_t = zeros(img_row, img_col);
I_orn_l = zeros(img_row, img_col);
I_edge = zeros(img_row, img_col);
for row = 1:img_row
	for col = 1:img_col
        Ex = double(dx(row, col));
        Ey = double(dy(row, col));
        mag = sqrt(Ex^2+Ey^2);
        I_edge(row, col) = mag;
        if (mag == 0)
            Ex = 0.001; Ey = 0;
        end
        ang = atan(Ey/Ex)*180/pi;
        theta = 0; lumda = 0;
        if (Ex>=0)
            if (ang>=67.5)
                theta = 90;  lumda = 1;   % ( 90,+1)
            elseif (ang>=22.5)   
                theta = 45;  lumda = 1;   % ( 45,+1)
            elseif (ang>=-22.5)
                theta = 0;   lumda = 1;   % (  0,+1)
            elseif (ang>=-67.5)
                theta = 135; lumda = -1;  % (135,-1)
            else
                theta = 90;  lumda = -1;  % ( 90,-1)
            end
        else
            if (ang > 0)
                ang = ang - 180;
                if (ang>=-112.5)
                    theta = 90;  lumda = -1;  % ( 90,-1)
                elseif (ang>=-157.5)
                    theta = 45;  lumda = -1;  % ( 45,-1)
                else
                    theta = 0;   lumda = -1;  % (  0,-1)
                end
            else
                ang = ang + 180;
                if (ang>=157.5)
                    theta = 0;   lumda = -1;  % (  0,-1)
                elseif (ang>=112.5)
                    theta = 135; lumda =  1;  % (135,+1)
                else
                    theta = 90;  lumda = 1;   % ( 90,+1)
                end  
            end
        end
        I_orn_t(row, col) = theta;
        I_orn_l(row, col) = lumda;
	end
end
toc;


%zero padding
pad_row = floor(win_row/2);
pad_col = floor(win_col/2);
I_edge_pad = padarray(I_edge, [pad_row pad_col], 0);
I_orn_t_pad = padarray(I_orn_t, [pad_row pad_col], 0);
I_orn_l_pad = padarray(I_orn_l, [pad_row pad_col], 0);

tic;
for row = 1:img_row
	for col = 1:img_col
        % window index
        win_row_idx = row:row+2*pad_row;
        win_col_idx = col:col+2*pad_col;
        
        % 1. calc "edge dense"
        win = I_edge_pad(win_row_idx, win_col_idx);
        I_dns(row,col) = sum(win(:));
        
        % 2. calc "edge orientation variance"
        win_t = I_orn_t_pad(win_row_idx, win_col_idx);
        win_l = I_orn_l_pad(win_row_idx, win_col_idx);
        N_d = nnz(win); % N(delta)
        for t=0:45:135
            sum_l = nnz(win_l==1 & win_t==t) + nnz(win_l==-1 & win_t==t);
            I_eov(row,col) = I_eov(row,col) - ((4*sum_l-N_d)/(3*N_d))^2;
        end
        
        % 3. calc "opposite edge pairs"
        for t=0:45:135
            phi_abs = abs(nnz(win_l==1 & win_t==t) - nnz(win_l==-1 & win_t==t));
            phi_sum = nnz(win_l==1 & win_t==t) + nnz(win_l==-1 & win_t==t);
            if phi_sum == 0
                f_oep = t3;
            else
                f_oep = -phi_abs / phi_sum;
            end
            I_oep(row,col) = I_oep(row,col) + f_oep;
        end
	end
end
toc;

% Text confidence map
I_tcm1 = I_dns.*exp(I_eov);
I_tcm2 = I_dns.*exp(I_eov+I_oep);

subplot(1,4,1);
imagesc(I);
subplot(1,4,2);
imagesc(I_dns);
subplot(1,4,3);
imagesc(I_tcm1);
subplot(1,4,4);
imagesc(I_tcm2);

%%%%%%%%%%%%%%%% Evaluation %%%%%%%%%%%%%
close all;
n=2:5; % Range of N in OTSU algo

subplot(1,size(n,2)+2,1);
imagesc(I);
title(['Image'])
subplot(1,size(n,2)+2,2);
imagesc(I_tcm2);
title(['TCM']);
for i=n
    IDX = otsu(I_tcm2,i);
    subplot(1,size(n,2)+2,i-n(1)+3);
    imagesc(IDX);
    title(['n = ' int2str(i)]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


T1=load('D:\Poyang\matsamples\patchdata\T1patches_sameneighbor.mat');%
S1 = T1.T1;%
T2=load('D:\Poyang\matsamples\patchdata\T2patches_sameneighbor.mat');%
S2 = T2.T2;%
yt = load('D:\Poyang\matsamples\patchdata\truelabel.mat');
ytrue = yt.y2;%
ytrue = ytrue';
pathname = 'D:\Poyang\matsamples\patchdata\Samples_pca_patch_truelabel_sameneighbor\';

for m = 1:40000
    k = zeros(30,30,9);
    for i = 1:3
        for j = 1:3
            patch1 = S1(m,i,j,:);
            patch1 = reshape(patch1,1,30);
            patch1 = patch1';
            patch2 = S2(m,i,j,:);
            patch2 = reshape(patch2,1,30);
            patch2 = patch2';
            a = 1 - (patch1 - patch2) * pinv(patch2);
            index = (i - 1) * 3 + j;
            k(:,:,index) = a;
        end
    end
    filename = strcat('Samples_',num2str(m), '_',num2str(ytrue(m,1)),'.mat');
    save([pathname,filename],'k');
end
            
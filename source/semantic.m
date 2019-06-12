
color = dir('../mapillary/color/');
color = color(3:end); 

color_label = load('../mat_files/color_label.mat');color_label=color_label.color_label;
labels = load('../mat_files/labels.mat');labels = labels.labels;
[color_label,ia,ic]=unique(color_label,'rows');
labels=labels(ia);

labels{1}='Unlabeled';
labels{12}='Bike Rack';

imgs_hist = zeros(numel(imgs),size(color_label,1)); nsteps=3;
for i=1:numel(imgs)
    try
        [X1, map1] = imread(strcat(color(i).folder,'/',imgs(i).name(1:end-4),'.png'));
    catch, filter1(i)=1;continue
    end
    im = ind2rgb(X1, map1)*255;
    for j=1:size(color_label,1)
        pixels = im(:,:,1)==color_label(j,1) & im(:,:,2)==color_label(j,2) & im(:,:,3)==color_label(j,3);
        imgs_hist(i,j) = sum(sum(pixels));        
    end
end

nofilter_imgs_hist = imgs_hist;

binaries = imgs_hist; binaries(binaries<1000)=0; 
binaries=binaries./binaries; binaries(isnan(binaries))=0;

%labels renaming, class merging and deleting
labels(strcmp(labels,'Phone Booth'))={'Car'};
labels(strcmp(labels,'Lane Marking - Crosswalk'))={'Crosswalk'};
labels(strcmp(labels,'Lane Marking - General'))={'Lane Marking'};
labels(strcmp(labels,'Bicyclist'))={'Rider'};
labels(strcmp(labels,'Barrier'))={'Barrier/Wall'};
labels(strcmp(labels,'Guard Rail'))={'Guard Rail/Trash Can'};

labels_filtered = labels;
index1 = find(strcmp(labels_filtered,'Curb')); index2 = find(strcmp(labels_filtered,'Curb Cut'));
imgs_hist(:,index1) = imgs_hist(:,index1) + imgs_hist(:,index2); 
binaries(:,index1) = binaries(:,index1) | binaries(:,index2); 
binaries(:,index2)=[]; imgs_hist(:,index2)=[]; labels_filtered(index2)=[]; 

index1 = find(strcmp(labels_filtered,'Traffic Sign (Back)')); index2 = find(strcmp(labels_filtered,'Traffic Sign Frame'));
imgs_hist(:,index1) = imgs_hist(:,index1) + imgs_hist(:,index2); 
binaries(:,index1) = binaries(:,index1) | binaries(:,index2);
binaries(:,index2)=[]; imgs_hist(:,index2)=[]; labels_filtered(index2)=[]; 
labels_filtered(strcmp(labels_filtered,'Traffic Sign (Back)'))={'Traffic Sign'};

index1 = find(strcmp(labels_filtered,'Terrain')); index2 = find(strcmp(labels_filtered,'Mountain'));
imgs_hist(:,index1) = imgs_hist(:,index1) + imgs_hist(:,index2); 
binaries(:,index1) = binaries(:,index1) | binaries(:,index2);
binaries(:,index2)=[]; imgs_hist(:,index2)=[]; labels_filtered(index2)=[]; 
labels_filtered(strcmp(labels_filtered,'Terrain'))={'Terrain/Mountain'};

todelete = [find(strcmp(labels_filtered,'Crosswalk - Plain')) find(strcmp(labels_filtered,'Bird')) find(strcmp(labels_filtered,'Traffic Sign (Front)')) find(strcmp(labels_filtered,'Sand')) find(strcmp(labels_filtered,'Billboard'))];
binaries(:,todelete)=[]; imgs_hist(:,todelete)=[]; labels_filtered(todelete)=[]; 

nclasses = size(labels_filtered,1);


io_data = readtable('../csv_files/io_results.csv');

dataDir = '../';
imgs = dir('../images'); [~,idimgs]=sort({imgs.name}); imgs=imgs(idimgs(3:end));
N = length(imgs);

filter = zeros(numel(imgs),1);
k=1;

% filtering dark and indoors images
for i=1:N
    % filter dark images
    img2 = imread(strcat(imgs(i).folder,'/',imgs(i).name));
    img = rgb2lab(img2); 
    if mean(mean(img(:,:,1)))<50
        filter(i)=1;
        continue   
    end

    % filter indoors images
    name_match = strcat('../images/',imgs(i).name);
    if io_data.Var2(find(strcmp(io_data.Var1(:),name_match))) == 0
        filter(i)=1;
        continue
    end  
end

imgs(find(filter))=[];
filter1 = zeros(numel(imgs),1);



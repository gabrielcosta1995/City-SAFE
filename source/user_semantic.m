
user_stats = zeros(5000,5);
j=0;jj=1;nclasses=size(imgs_hist,2);
semantics = binaries;

semantic_user = zeros(5000,2*nclasses);
user_frequencies = zeros(5000,2,2,nclasses);
evolution = zeros(5000,2*nclasses);
cookie = zeros(5000,1);

for i=1:size(comparisons,1)
      
    index = find(user_stats(:,1)==comparisons{i,'Var2'}); 
    if isempty(index)      
        j=j+1; index = j; 
        user_stats(j,1) = comparisons{i,'Var2'};
        user_stats(j,2) = 0;
    end
    
    image1 = char(comparisons{i,'Var8'}); image1 = image1(65:end-4);
    img1_index = find(strcmp(fulldataset_id,image1));
    image2 = char(comparisons{i,'Var9'}); image2 = image2(65:end-4);
    img2_index = find(strcmp(fulldataset_id,image2));
    
    if isempty(img1_index) || isempty(img2_index)
        continue
    end
    
    if comparisons{i,'Var10'} == -1
        user_stats(index,2) = user_stats(index,2)+1;
        user_stats(index,3) = user_stats(index,3) + 1;
        semantic_user(index,1:nclasses) = semantic_user(index,1:nclasses) + semantics(img1_index,:);
        semantic_user(index,nclasses+1:end) = semantic_user(index,nclasses+1:end) + semantics(img2_index,:);
        evolution(i,1:nclasses) = semantics(img1_index,:);
        evolution(i,1+nclasses:end) = semantics(img2_index,:);
        cookie(i) = comparisons{i,'Var2'};
        
        diff = semantics(img1_index,:)-semantics(img2_index,:);
        temp = diff; temp(temp<0)=0;
        user_frequencies(index,1,2,:) = user_frequencies(index,1,2,:)+reshape(temp,[1,1,1,size(temp,2)]);
        temp = -diff; temp(temp<0)=0;
        user_frequencies(index,2,1,:) = user_frequencies(index,2,1,:)+reshape(temp,[1,1,1,size(temp,2)]);
        temp = semantics(img1_index,:)+semantics(img2_index,:);temp(temp<2)=0;temp(temp==2)=1;
        user_frequencies(index,1,1,:) = user_frequencies(index,1,1,:)+reshape(temp,[1,1,1,size(temp,2)]);
        temp = semantics(img1_index,:)+semantics(img2_index,:);temp(temp>0)=-1;temp=temp+1;
        user_frequencies(index,2,2,:) = user_frequencies(index,2,2,:)+reshape(temp,[1,1,1,size(temp,2)]);
        
    elseif comparisons{i,'Var10'} == 1
        user_stats(index,2) = user_stats(index,2)+1;
        user_stats(index,5) = user_stats(index,5) + 1;
        semantic_user(index,1:nclasses) = semantic_user(index,1:nclasses) + semantics(img2_index,:);
        semantic_user(index,nclasses+1:end) = semantic_user(index,nclasses+1:end) + semantics(img1_index,:);
        evolution(i,1:nclasses) = semantics(img2_index,:);
        evolution(i,1+nclasses:end) = semantics(img1_index,:);
        cookie(i) = comparisons{i,'Var2'};

        diff = semantics(img2_index,:)-semantics(img1_index,:);
        temp = diff; temp(temp<0)=0;
        user_frequencies(index,1,2,:) = user_frequencies(index,1,2,:)+reshape(temp,[1,1,1,size(temp,2)]);
        temp = -diff; temp(temp<0)=0;
        user_frequencies(index,2,1,:) = user_frequencies(index,2,1,:)+reshape(temp,[1,1,1,size(temp,2)]);
        temp = semantics(img1_index,:)+semantics(img2_index,:);temp(temp<2)=0;temp(temp==2)=1;
        user_frequencies(index,1,1,:) = user_frequencies(index,1,1,:)+reshape(temp,[1,1,1,size(temp,2)]);
        temp = semantics(img1_index,:)+semantics(img2_index,:);temp(temp>0)=-1;temp=temp+1;
        user_frequencies(index,2,2,:) = user_frequencies(index,2,2,:)+reshape(temp,[1,1,1,size(temp,2)]);
        
    elseif comparisons{i,'Var10'} == 0
        user_stats(index,4) = user_stats(index,4) + 1;
        continue
    end
end

user_frequencies = user_frequencies(1:j,:,:,:);
semantic_user = semantic_user(1:j,:);
user_stats = user_stats(1:j,:);
evolution(cookie==0,:)=[];
cookie(cookie==0)=[];


    
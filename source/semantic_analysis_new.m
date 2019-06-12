colors_rgb = [[217/255,83/255,25/255];[126/255,47/255,142/255];[237/255,177/255,32/255];[0,114/255,189/255];[0,126/255,70/255];[0,1,1];[0,0,0]];
colors = [{'r'};{'g'};{'b'};{'y'};{'m'};{'c'};{'k'}];

%only active users
filtered_sem_user = semantic_user(find(user_stats(:,2)>10),:); filtered_user_frequencies = user_frequencies(find(user_stats(:,2)>10),:,:,:); filtered_stats = user_stats(find(user_stats(:,2)>10),:); filtered_labels=labels_filtered;


%only classes that appeared enough
allocurrences = filtered_sem_user(:,1:nclasses)+filtered_sem_user(:,nclasses+1:end);
temp = [find(sum(allocurrences,1)<200)]; 
temp2 = [temp temp+nclasses];
filtered_sem_user(:,temp2)=[]; 
filtered_user_frequencies(:,:,:,temp)=[]; 
filtered_labels(temp)=[];
nclasses_filtered = 0.5*size(filtered_sem_user,2); nusers = size(filtered_sem_user,1);
user_frequencies_stacked= user_frequencies; user_frequencies_stacked(:,:,:,temp)=[];

filtered_evolution=evolution;
filtered_evolution(:,temp2)=[];

%all data by user
all_data_wl = zeros(nusers,2,nclasses_filtered);
for i=1:nusers
    all_data_wl(i,1,:) = filtered_sem_user(i,1:nclasses_filtered)/sum(sum(filtered_sem_user));
    all_data_wl(i,2,:) = filtered_sem_user(i,1+nclasses_filtered:end)/sum(sum(filtered_sem_user));
end
all_data_2n = cat(2,reshape(all_data_wl(:,1,:),[nusers,nclasses_filtered]),reshape(all_data_wl(:,2,:),[nusers,nclasses_filtered]));


%aggl hier clus
figure
cutoff=0.08;
Z = linkage(filtered_sem_user,'complete','cosine');
[H,T,perm]=dendrogram(Z,60,'ColorThreshold',cutoff);
set(H,'LineWidth',1.5)


%get cluster roots
 j=0; a=Z(end-j,3); ngroups=1; roots=[size(Z,1)+nusers];
while a>cutoff
    if ismember(size(Z,1)-j+nusers,roots)
        roots(roots==size(Z,1)-j+nusers)=[];
    end
    roots = [roots Z(end-j,1) Z(end-j,2)];
    ngroups = ngroups + 1;j=j+1;
    a=Z(end-j,3);
end

%assign user to cluster
cluster_id = nan(nusers,1); group_users = zeros(ngroups,1);
for i=1:numel(roots)
    new_group=[roots(i)];
    while max(new_group)>nusers
        temp=max(new_group); new_group(new_group==temp)=[];
        new_group=[new_group Z(temp-nusers,1) Z(temp-nusers,2)];
    end
%     centroids(i,:)=sum(filtered_sem_user(new_group,:),1)/numel(new_group); 
    group_users(i) = numel(new_group); cluster_id(new_group)=i;
end

%all data by user cluster
all_cluster_data_wl = zeros(ngroups,2,nclasses_filtered);
cluster_freqs = zeros(ngroups,2,2,nclasses_filtered);
for i=1:ngroups
    cluster = sum(filtered_sem_user(cluster_id==i,:),1);
    all_cluster_data_wl(i,1,:) = cluster(1:nclasses_filtered)/sum(sum(filtered_sem_user));
    all_cluster_data_wl(i,2,:) = cluster(1+nclasses_filtered:end)/sum(sum(filtered_sem_user));
    cluster_freqs(i,:,:,:) = sum(filtered_user_frequencies(cluster_id==i,:,:,:),1);
end
all_cluster_data_2n = cat(2,reshape(all_cluster_data_wl(:,1,:),[ngroups,nclasses_filtered]),reshape(all_cluster_data_wl(:,2,:),[ngroups,nclasses_filtered]));


%labels for loss classes
filtered_labels_loss = filtered_labels;
for i=1:nclasses_filtered
    filtered_labels_loss{i}=strcat(filtered_labels_loss{i},'_l');
end

cluster_probs = bsxfun(@rdivide,cluster_freqs,sum(sum(sum(cluster_freqs,4),3),2));windata=[];lossdata=[];
%% pca of matrix A -> A_ij = P(class_j|user_i) to neutralize #votes/user
A = bsxfun(@rdivide,all_data_2n,sum(all_data_2n,2));
[pc, latent]=pca(zscore(A),size(A,2)); latent = latent./sum(sum(latent));
semantic_pca = zscore(A)*pc;
figure
scatter(semantic_pca(:,1),semantic_pca(:,2),50,colors_rgb(cluster_id,:),'filled');
title('pca 2d')
figure
scatter3(semantic_pca(:,1),semantic_pca(:,2),semantic_pca(:,3),50,colors_rgb(cluster_id,:),'filled');
title('pca 3d')

%% isomap of A
D = zeros(nusers);
for i1=1:nusers
    for i2=i1:nusers
        D(i1,i2)=1 - dot(A(i1,:),A(i2,:))/(norm(A(i1,:))*norm(A(i2,:)));
        D(i2,i1)=1 - dot(A(i1,:),A(i2,:))/(norm(A(i1,:))*norm(A(i2,:)));
    end
end
% [Y, R, E] = Isomap(D,'k',5);

%% t-sne of A
ydata = tsne(A);
scatter(ydata(:,1),ydata(:,2),50,colors_rgb(cluster_id,:),'filled')
title('t-sne')

%% P(c,w,l|u)

for i=1:ngroups
    figure
    plotdata = [reshape(cluster_probs(i,1,2,:),[1,nclasses_filtered]);reshape(cluster_probs(i,2,1,:),[1,nclasses_filtered])];
    bar(categorical(filtered_labels,filtered_labels),transpose(plotdata));
    title(strcat('P(winner,~loser,class|cluster',' ',num2str(i),'); P(~winner,loser,class|cluster',' ',num2str(i),') -> #users=',num2str(group_users(i)),'; #comparisons=',num2str(sum(filtered_stats(cluster_id==i,2)))));
    legend('winner,~loser','~winner,loser')
    windata = [windata;reshape(cluster_probs(i,1,2,:),[1,nclasses_filtered])];
    lossdata = [lossdata;reshape(cluster_probs(i,2,1,:),[1,nclasses_filtered])];
end
figure
bar(categorical(filtered_labels,filtered_labels),transpose(windata));
title(strcat('P(winner,~loser,class|cluster)'));
legend('cluster 1 -> 2 users;22 comparisons','cluster 2 -> 20 users;364 comparisons','cluster 3 -> 101 users;3204 comparisons','cluster 4 -> 50 users;7326 comparisons','cluster 5 -> 24 users;533 comparisons')
figure
bar(categorical(filtered_labels,filtered_labels),transpose(lossdata));
title(strcat('P(~winner,loser,class|cluster)'));
legend('cluster 1 -> 2 users;22 comparisons','cluster 2 -> 20 users;364 comparisons','cluster 3 -> 101 users;3204 comparisons','cluster 4 -> 50 users;7326 comparisons','cluster 5 -> 24 users;533 comparisons')
figure
bar(categorical(filtered_labels,filtered_labels),transpose(reshape(cluster_probs(:,1,1,:),[ngroups,nclasses_filtered])));
title(strcat('P(winner,loser,class|cluster)'));
legend('cluster 1 -> 2 users;22 comparisons','cluster 2 -> 20 users;364 comparisons','cluster 3 -> 101 users;3204 comparisons','cluster 4 -> 50 users;7326 comparisons','cluster 5 -> 24 users;533 comparisons')

%% P(w,l|c,u)
% temp = sum(cluster_probs,1); 
temp = cluster_probs;
myprobs = cluster_probs./(temp(:,1,1,:)+temp(:,1,2,:)+temp(:,2,1,:)); windata=[];lossdata=[];
for i=1:ngroups
    if i==1 || i==3 || i==6
        continue
    end
    plotdata = [reshape(myprobs(i,1,2,:),[1,nclasses_filtered]);reshape(myprobs(i,2,1,:),[1,nclasses_filtered])];
%     figure
%     bar(categorical(filtered_labels,filtered_labels),transpose(plotdata));
%     title(strcat('P(winner,~loser|class,cluster',' ',num2str(i),'); P(~winner,loser|class,cluster',' ',num2str(i),') -> #users=',num2str(group_users(i)),'; #comparisons=',num2str(sum(filtered_stats(cluster_id==i,2)))));
%     legend('P(W,~L|C,U)','P(~W,L|C,U)')
    windata = [windata;reshape(myprobs(i,1,2,:),[1,nclasses_filtered])];
    lossdata = [lossdata;reshape(myprobs(i,2,1,:),[1,nclasses_filtered])];
end
figure
bar(categorical(filtered_labels(plotorder),filtered_labels(plotorder)),transpose(windata(:,plotorder)));
%title(strcat('P(winner,~loser|class,cluster)'));
title(strcat('$P(W,\neg L|S,U)$'),'interpreter','latex');
legend('cluster 2','cluster 4','cluster 5')
figure
bar(categorical(filtered_labels(plotorder),filtered_labels(plotorder)),transpose(lossdata(:,plotorder)));
%title(strcat('P(~winner,loser|class,cluster)'));
title(strcat('$P(\neg W,L|S,U)$'),'interpreter','latex');
legend('cluster 2','cluster 4','cluster 5')

%% P(C|w,l,U)
% class_composition = zeros(ngroups,2,nclasses_filtered);class_comp_w_all = []; class_comp_l_all=[];
class_composition = cluster_probs./sum(cluster_probs,4);windata=[];lossdata=[];
for i=1:ngroups
%     class_composition(i,1,:) = cluster_probs(i,1,2,1:nclasses_filtered)./sum(cluster_probs(i,,2,1:nclasses_filtered));
%     class_composition(i,2,:) = cluster_A(i,1+nclasses_filtered:end)/sum(cluster_A(i,1+nclasses_filtered:end));
    figure
    bar(categorical(filtered_labels,filtered_labels),transpose([reshape(class_composition(i,1,2,:),[1,nclasses_filtered]);reshape(class_composition(i,2,1,:),[1,nclasses_filtered])]));
    title(strcat('P(class|winner,~loser,cluster',' ',num2str(i),'); P(class|~winner,loser,cluster',' ',num2str(i),') -> #users=',num2str(group_users(i)),'; #comparisons=',num2str(sum(filtered_stats(cluster_id==i,2)))));
    legend('winner,~loser','~winner,loser')
    windata = [windata;reshape(class_composition(i,1,2,:),[1,nclasses_filtered])];
    lossdata = [lossdata;reshape(class_composition(i,2,1,:),[1,nclasses_filtered])];
end
figure
bar(categorical(filtered_labels,filtered_labels),transpose(windata));
title(strcat('P(class|winner,~loser,cluster)'));
legend('cluster 1 -> 2 users;22 comparisons','cluster 2 -> 20 users;364 comparisons','cluster 3 -> 101 users;3204 comparisons','cluster 4 -> 50 users;7326 comparisons','cluster 5 -> 24 users;533 comparisons')
figure
bar(categorical(filtered_labels,filtered_labels),transpose(lossdata));
title(strcat('P(class|~winner,loser,cluster)'));
legend('cluster 1 -> 2 users;22 comparisons','cluster 2 -> 20 users;364 comparisons','cluster 3 -> 101 users;3204 comparisons','cluster 4 -> 50 users;7326 comparisons','cluster 5 -> 24 users;533 comparisons')

%% new plots stacked bars
temp = sum(cluster_freqs,1)./sum(sum(sum(cluster_freqs,1),2),3);
temp = sum(cluster_probs,1); 
% temp=cluster_probs; i=6;
temp = sum(user_frequencies_stacked,1);
new_plots = temp./(temp(:,1,1,:)+temp(:,1,2,:)+temp(:,2,1,:)); %new_plots=new_plots(i,:,:,:);
new_plots1 = new_plots(1,1,1,:);%+new_plots(1,2,2,:);

newplotdata = [reshape(new_plots(1,1,2,:),[1,nclasses_filtered]);reshape(new_plots(1,2,1,:),[1,nclasses_filtered]);reshape(new_plots1,[1,nclasses_filtered])];
figure
bar(categorical(filtered_labels(plotorder),filtered_labels(plotorder)),transpose(newplotdata(:,plotorder)),'stacked');
legend('P(W,\negL|C)','P(\negW,L|C)','P(W,L|C)','interpreter','latex')
%title(strcat('Cluster ', num2str(i)))

%% new plots evolution
u_star = sum(filtered_sem_user,1);
% u_star = all_cluster_data_2n(4,:);
figure
for i=1:ngroups
    [maxim,ind]= max(filtered_stats(cluster_id==i,2));
    temp = filtered_stats(cluster_id==i);
    bestcookie = temp(ind);
    cookie_comps = filtered_evolution(cookie==bestcookie,:);
    plot_evol = zeros(maxim,2*nclasses_filtered);
    distance_evol = zeros(maxim,1);
    for ii=1:maxim
        plot_evol(ii,:)=sum(cookie_comps(1:ii,:),1);
        distance_evol(ii) = 1-dot(u_star,plot_evol(ii,:))/(norm(u_star)*norm(plot_evol(ii,:)));
    end    
    plot(1:maxim,distance_evol)
    hold on
end
legend('cluster 1','cluster 2','cluster 3','cluster 4','cluster 5','cluster 6')
ylabel('Cosine error to $\tilde{c}$','Interpreter','latex')
xlabel('\#comparisons','Interpreter','latex')

        
    


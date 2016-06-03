clear;clc;close all;
load('../datasets/yale/Yale_32x32.mat');
addpath(genpath('../helpers'))
%% pca ��ά
full_feature=fea;
[COEFF, SCORE, LATENT] = pca(full_feature);
index = pca_percent(LATENT,0.9);
X = SCORE(:,1:index);
X = X';%��������Ϣת����ÿһ��Ϊһ������.
% plot(X(1,1:11),X(2,1:11),'bo');
% hold ;
% plot(X(1,12:22),X(2,12:22),'ro');
X = zscore(X,1,2);


%% һЩ�������趨�ͳ���ֵ
class_samples_num = 11;
all_samples_num = size(X,2);
class_num = all_samples_num/class_samples_num;
class_tag = 1;
class_X =X(:,1:class_samples_num);
t_collec = [];
h_collec = [];
delta = 1.0e-3;

%% ��ÿ�����������ŵ�t
for i = 1 : all_samples_num
    if(mod(i,class_samples_num)==1 && i ~= 1)%������������仯ʱ
        class_tag = class_tag + 1;%���������
        class_X = X(:,(class_tag-1)*class_samples_num+1:class_tag*class_samples_num);%���¸��������� 
    end
    
    sample_class_index = mod(i-1,class_samples_num);% �����ڸ����еĽű�0-10
    class_X_copy = class_X; %���ݵ����������Ϣ
    class_X(:,sample_class_index+1) = []; %�ڵ�ǰ���������޳�������Ϊѵ������
 
%     X(:,i)
%    class_X

  
  tmp_t = opt_t_desp(X(:,i),class_X);
  %����õ�t Ԫ���м���0Ԫ��
  if(sample_class_index ==0)
      t = [0;tmp_t(sample_class_index+1:end,:)];
  elseif(sample_class_index == class_samples_num-1)
      t = [tmp_t(1:end,:);0];
  else    
      t = [tmp_t(1:sample_class_index,:);0;tmp_t(sample_class_index+1:end,:)];
  end    
  pre_t = zeros(floor((i-1)/class_samples_num)*class_samples_num,1);
  after_t = zeros((all_samples_num/class_samples_num-ceil((i)/class_samples_num))*class_samples_num,1);
  h = [pre_t;t;after_t];%��t��Ϊh
  class_X = class_X_copy;%�ָ����ݵ����������Ϣ
  t_collec = [t_collec t];
  h_collec = [h_collec h];
  
end

%% ��ÿ������������ŵ�s
% s_collec = [];
% for i = 1 : all_samples_num
%     %clear ����
%     ctphixij = [];
%     ctc = [];
%     ctci = [];
%     class_tag =  floor((i-1)/class_samples_num); %��0��ʼ��
%     tmp_X = X;
%     tmp_X(:,class_tag * class_samples_num +1 : (class_tag+1)* class_samples_num) =[];
%     tmp_s = opt_s_desp(X(:,i),X,tmp_X,h_collec(:,i),delta);
%     zero_vacant = zeros(class_samples_num,1);
%     if(class_tag ==0)
%         s = [zero_vacant;tmp_s];
%     elseif(class_tag == class_num-1)
%         s = [tmp_s;zero_vacant];
%     else   
%         s = [tmp_s(1:(class_tag+1)* class_samples_num,:);zero_vacant;tmp_s((class_tag+1)* class_samples_num+1:end,:)];
%     end
%     s_collec = [s_collec s];
% end
%% �����������ֵ
D = h_collec;
M = D + D' + D'* D;
T1 = X * X';
T2 = X * M * X';
T = T2\T1;
[VA,VC]=eig(T);
reducedVector = VA * X ;
% reducedVector_train =VA *X_train;
desp_res = knn(reducedVector, reducedVector, gnd', 5);
pca_res = knn(X, X, gnd', 5)







function [eigVector,eigValue,reducedVector] = desp(x_train,y_train,option)
%desp������ϡ�豣��Ƕ��
%   Input:
%         x_train: train data, P by N matrix, each column is a sample;
%         y_train: class label, 1 by N matrix
%         option:
%                detal: 

N = size(x_train,2);
same_class_samples; %�����ͬ������������
diff_class_samples; %��Ų�ͬ������������
for i = 1 : N
    same_class_samples = [];
    diff_class_samples = [];
    same_class_index = y_train==y_train(i)
end
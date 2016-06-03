function [eigVector,eigValue,reducedVector] = desp(x_train,y_train,option)
%desp：鉴别稀疏保持嵌入
%   Input:
%         x_train: train data, P by N matrix, each column is a sample;
%         y_train: class label, 1 by N matrix
%         option:
%                detal: 

N = size(x_train,2);
same_class_samples; %存放相同类别的向量数据
diff_class_samples; %存放不同类别的向量数据
for i = 1 : N
    same_class_samples = [];
    diff_class_samples = [];
    same_class_index = y_train==y_train(i)
end
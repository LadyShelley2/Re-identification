function rate = correct_rate(X_test,Y_test, X_train, y_train, K)
y = knn(X_test, X_train, y_train, K);
amount =size(X_test,2);
rate = sum(sum(y==Y_test))/amount;
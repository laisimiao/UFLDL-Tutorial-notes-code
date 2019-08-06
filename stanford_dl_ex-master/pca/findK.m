he = 0;
for i=1:length(lambda)
he = he + lambda(i);
per = he / sum_lambda;
    if per >=0.90
        fprintf('now i is:%d\n',i);
        break;
    end
end
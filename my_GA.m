clc
clear
close all

dimention_range = [-5 5; -5 5];
pool_size = 30;

maxgen = 100;

pool = generate_pool(50, dimention_range);
best_kromosom_in_each_gen = zeros(maxgen,2);
best_fitness_in_each_gen = zeros(maxgen,1);

for gen = 1:maxgen
    for k = 1:floor(pool_size/2)
        selected_kromosoms = selection(pool, 0);
        
        A = pool(selected_kromosoms(1),:);
        B = pool(selected_kromosoms(2),:);
        
        X = pheno2geno(A, dimention_range);
        Y = pheno2geno(B, dimention_range);
        
        r = rand(1);
        if r > 0.9
            X = bin_mutation(X);
            Y = bin_mutation(Y);
        end
        [X_prim, Y_prim] = bin_crossover(X, Y);
        
        A_prim = geno2pheno(X_prim, dimention_range);
        B_prim = geno2pheno(Y_prim, dimention_range);
        
        if fitness(A) < fitness(A_prim)
            pool(selected_kromosoms(1),:) = A_prim;
        end
        if fitness(B) < fitness(B_prim)
            pool(selected_kromosoms(2),:) = B_prim;
        end
    end
    [best_kromosom_in_each_gen(gen,:), best_fitness_in_each_gen(gen)] = find_best_kromosom(pool);
    
    figure(2);
    plot(1:maxgen, log(best_fitness_in_each_gen));
    grid minor;
    xlabel('gen');
    ylabel('fitness');
    title('best kromosom')
    drawnow;

end

[best_fitness, best_pos] = max(best_fitness_in_each_gen);
best_kromosom = best_kromosom_in_each_gen(best_pos, :);

disp(best_kromosom);






function [best_kromosom, best_fitness] = find_best_kromosom(pool)
pool_size = size(pool, 1);

fitness_array = zeros(1, pool_size);
for k = 1:pool_size
    pheno_X = pool(k,:);
    fitness_array(k) = fitness(pheno_X);
end
[best_fitness, best_pos] = max(fitness_array);
best_kromosom = pool(best_pos, :);
end

function [selected_kromosoms] = selection(pool,show_fitness)
pool_size = size(pool, 1);

fitness_array = zeros(1, pool_size);
for k = 1:pool_size
    pheno_X = pool(k,:);
    fitness_array(k) = fitness(pheno_X);
end

sum_of_fitness = sum(fitness_array);
fitness_pdf = fitness_array / sum_of_fitness;
fitness_cdf = cumsum(fitness_pdf);

r = rand(2,1);
[~, selected_kromosoms] = max(r <= fitness_cdf,[],2);

if show_fitness
    figure(show_fitness);
    clf;
    bar(fitness_cdf);
    drawnow;
end
end

function [pool] = generate_pool(pool_size, kromosom_range)
kromosom_dimention = size(kromosom_range, 1);
pool = rand(pool_size,kromosom_dimention);

min_range = kromosom_range(:,1)';
max_range = kromosom_range(:,2)';
range_dist = max_range - min_range;

pool = range_dist .* pool + min_range;
end

function [fitness_value] = fitness(pheno_X)
evaluate_value = evaluate(pheno_X);

fitness_value = evaluate_value;
end

function [evaluate_value] = evaluate(pheno_X)
x = pheno_X(1);
y = pheno_X(2);
z = 1/(x^2+eps)+1/(y^2+eps);
evaluate_value = z;
end

function [geno_X] = pheno2geno(pheno_X, pheno_range)
GA = bin_GA;

% normalized in dec length
max_dec = 2^GA.code_length-1;
pheno_min_in_range = pheno_range(:,1)';
pheno_range_length = pheno_range(:,2)' - pheno_range(:,1)';
normalized_pheno_X = round( (pheno_X - pheno_min_in_range) ./ pheno_range_length * max_dec);

pheno_length = length(normalized_pheno_X);

geno_X = blanks(GA.code_length*pheno_length);
for i = 1:length(normalized_pheno_X)
    geno_X((i-1)*GA.code_length + 1:i*GA.code_length) = bin_coding(normalized_pheno_X(i));
end
end

function [pheno_X] = geno2pheno(geno_X, pheno_range)
GA = bin_GA;
pheno_length = length(geno_X)/GA.code_length;

pheno_X = zeros(1,pheno_length);
for i = 1:pheno_length
    pheno_X(i) = bin_decoding(geno_X((i-1)*GA.code_length + 1:i*GA.code_length));
end

% retrieve normal length
max_dec = 2^GA.code_length - 1;
pheno_min_in_range = pheno_range(:,1)';
pheno_range_length = pheno_range(:,2)' - pheno_range(:,1)';
pheno_X = pheno_X/max_dec.*pheno_range_length+pheno_min_in_range;
end

function [X_prim, Y_prim] = bin_crossover(X, Y)
len_X = length(X)-1;
k = 1+round(len_X * rand(1,2));
k = sort(k);
X_prim = [X(1:k(1)) Y(k(1)+1:k(2)) X(k(2)+1:end)];
Y_prim = [Y(1:k(1)) X(k(1)+1:k(2)) Y(k(2)+1:end)];
end

function [X] = bin_mutation(X)
len_X = length(X)-1;
k = 1+round(len_X * rand(1));
if X(k) == '1'
    X(k) = '0';
else
    X(k) = '1';
end
end

function [bin_num] = bin_coding(dec_num)
GA = bin_GA;

bin_num = dec2bin(dec_num, GA.code_length);
len_bin_num = length(bin_num);
bin_num = bin_num(max(len_bin_num-GA.code_length, 0) + 1 : end);
end

function [dec_num] = bin_decoding(bin_num)
dec_num = bin2dec(bin_num);
end

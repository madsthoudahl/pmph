
fact :: Integer -> Integer
fact 0 = 1
fact n = n * fact (n-1) 

fact2 :: Integer -> Integer
fact2 n = product [1..n]

dotproduct :: [Integer] -> [Integer] -> Integer
dotproduct [] _ys = 0
dotproduct _xs [] = 0
dotproduct (x:xs) (y:ys) = (x * y) + dotproduct xs ys

mulPairs :: [Integer] -> [Integer] -> [Integer] 
--mulPairs a b = map (\(x,y) -> x*y) (zip a b)
--mulPairs = zipWith (*)
mulPairs a b = zipWith (*) a b

dp2 :: [Integer] -> [Integer] -> Integer 
--dp2 a b = foldl (+) 0 (zipWith (*) a b)
dp2 a b = sum (zipWith (*) a b)

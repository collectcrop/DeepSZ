echo start > ./data/compression_ratios_fc1.txt
./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A 1E-3 -i ./data/fc1-data-o.dat -1 30720
./SZ/build/bin/sz -x -f -s ./data/fc1-data-o.dat.sz -1 30720
mv -f ./data/fc1-data-o.dat.sz.out ./data/fc1-data-1E-3.dat
echo 1E-3 >> ./data/compression_ratios_fc1.txt
wc -c ./data/fc1-data-o.dat.sz >> ./data/compression_ratios_fc1.txt
./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A 2E-3 -i ./data/fc1-data-o.dat -1 30720
./SZ/build/bin/sz -x -f -s ./data/fc1-data-o.dat.sz -1 30720
mv -f ./data/fc1-data-o.dat.sz.out ./data/fc1-data-2E-3.dat
echo 2E-3 >> ./data/compression_ratios_fc1.txt
wc -c ./data/fc1-data-o.dat.sz >> ./data/compression_ratios_fc1.txt
./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A 3E-3 -i ./data/fc1-data-o.dat -1 30720
./SZ/build/bin/sz -x -f -s ./data/fc1-data-o.dat.sz -1 30720
mv -f ./data/fc1-data-o.dat.sz.out ./data/fc1-data-3E-3.dat
echo 3E-3 >> ./data/compression_ratios_fc1.txt
wc -c ./data/fc1-data-o.dat.sz >> ./data/compression_ratios_fc1.txt
./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A 4E-3 -i ./data/fc1-data-o.dat -1 30720
./SZ/build/bin/sz -x -f -s ./data/fc1-data-o.dat.sz -1 30720
mv -f ./data/fc1-data-o.dat.sz.out ./data/fc1-data-4E-3.dat
echo 4E-3 >> ./data/compression_ratios_fc1.txt
wc -c ./data/fc1-data-o.dat.sz >> ./data/compression_ratios_fc1.txt
./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A 5E-3 -i ./data/fc1-data-o.dat -1 30720
./SZ/build/bin/sz -x -f -s ./data/fc1-data-o.dat.sz -1 30720
mv -f ./data/fc1-data-o.dat.sz.out ./data/fc1-data-5E-3.dat
echo 5E-3 >> ./data/compression_ratios_fc1.txt
wc -c ./data/fc1-data-o.dat.sz >> ./data/compression_ratios_fc1.txt
./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A 6E-3 -i ./data/fc1-data-o.dat -1 30720
./SZ/build/bin/sz -x -f -s ./data/fc1-data-o.dat.sz -1 30720
mv -f ./data/fc1-data-o.dat.sz.out ./data/fc1-data-6E-3.dat
echo 6E-3 >> ./data/compression_ratios_fc1.txt
wc -c ./data/fc1-data-o.dat.sz >> ./data/compression_ratios_fc1.txt
./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A 7E-3 -i ./data/fc1-data-o.dat -1 30720
./SZ/build/bin/sz -x -f -s ./data/fc1-data-o.dat.sz -1 30720
mv -f ./data/fc1-data-o.dat.sz.out ./data/fc1-data-7E-3.dat
echo 7E-3 >> ./data/compression_ratios_fc1.txt
wc -c ./data/fc1-data-o.dat.sz >> ./data/compression_ratios_fc1.txt
./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A 8E-3 -i ./data/fc1-data-o.dat -1 30720
./SZ/build/bin/sz -x -f -s ./data/fc1-data-o.dat.sz -1 30720
mv -f ./data/fc1-data-o.dat.sz.out ./data/fc1-data-8E-3.dat
echo 8E-3 >> ./data/compression_ratios_fc1.txt
wc -c ./data/fc1-data-o.dat.sz >> ./data/compression_ratios_fc1.txt
./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A 9E-3 -i ./data/fc1-data-o.dat -1 30720
./SZ/build/bin/sz -x -f -s ./data/fc1-data-o.dat.sz -1 30720
mv -f ./data/fc1-data-o.dat.sz.out ./data/fc1-data-9E-3.dat
echo 9E-3 >> ./data/compression_ratios_fc1.txt
wc -c ./data/fc1-data-o.dat.sz >> ./data/compression_ratios_fc1.txt

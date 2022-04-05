mkdir Release && cd Release
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd main/
./sayhello -print_times 10

cd ../test/
./main_test

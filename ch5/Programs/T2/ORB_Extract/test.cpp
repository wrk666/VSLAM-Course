#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <list>
#include <vector>
#include <execution>
#include <string>

using namespace std;

struct Sum{
    Sum() : sum{ 0 } { }
    void operator()(int n) { sum += n; }
    int sum;
};

int main()
{
    int arr[] = { 1,2,3,4,5 };
    std::vector<int> vec = { 11,12,13,14,15 };
    std::initializer_list<int> lst = { 21,22,23,24,25 };
    std::map<string, int> dict = { {"Tom",10},{"Bob",20} };

//    for_each(begin(arr), end(arr), [](int i)->void {cout << i << endl; });

    std::for_each(std::execution::par_unseq, begin(arr), end(arr), [] (int i) ->void {cout << i << endl; });


//    for_each(begin(vec), end(vec), [](int i)->void {cout << i << endl; });
//    for_each(begin(lst), end(lst), [](int i)->void {cout << i << endl; });
//    for_each(begin(dict), end(dict), [](std::pair<string, int> ele)->void {cout << ele.first .c_str()<< ele.second << endl; });
//
//    auto print = [](const int& n) { cout << n << endl; };
//    for_each(vec.begin(), vec.end(), print);
//
//    for_each(vec.begin(), vec.end(), [](int &n) { n++; });//元素+1
//    for_each(vec.begin(), vec.end(), print);
//
//    Sum s = std::for_each(vec.begin(), vec.end(), Sum());// 对每个数调用 Sum::operator()
//    cout << "sum: " << s.sum << '\n';
    return 0;
}
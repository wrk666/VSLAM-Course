//
// Created by wrk on 2022/3/19.
//

#include <iostream>
#include <string>
//#inlcude <boost>
#include <sstream>

using namespace std;

int main()
{
//    int aa = 1;
//    char str1[8];
////    string str1;
//    int length = sprintf(str1, "%X", aa);
//    cout<<length<<endl;
//    cout<<str1<<endl;

//    int aa = 30;
//    stringstream ss;
//    ss<<aa;
//    string s1 = ss.str();
//    cout<<s1<<endl; // 30
//
//    string s2;
//    ss>>s2;
//    cout<<s2<<endl; // 30

//    int aa = 30;
//    string s = boost::lexical_cast<string>(aa);
//    cout<<s<<endl; // 30

    int aa = 30;
    stringstream ss;
    ss<<aa;
    string s1 = ss.str();
    cout<<s1<<endl; // 30

    string s2;
    ss>>s2;
    cout<<s2<<endl; // 30
}

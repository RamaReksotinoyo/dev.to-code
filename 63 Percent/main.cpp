#include <iostream>
#include<bits/stdc++.h>

// using namespace std;

void solve(){
    double temp = 1.0;
    long n = 10000.0;
    // double arrSize = sizeof(arr)/sizeof(arr[0]);
    // std::cout<<arrSize<<std::endl;
    // for(int j=1; j<=arrSize; j++){
    float results = 1.0 - (1.0/n); 
    for(int i = 1; i<=n; i++){
        temp = temp * results;
    }
    double rslt = 1.0 - temp;
    double output = round(rslt);
    // std::cout<< std::setprecision(2) <<rslt<<std::endl;
    std::cout<<rslt<<std::endl;
    // }
}


float round(float var)
{
    // 37.66666 * 100 =3766.66
    // 3766.66 + .5 =3767.16    for rounding off value
    // then type cast to int so value is 3767
    // then divided by 100 so the value converted into 37.67
    float value = (int)(var * 100 + .5);
    return (float)value / 100;
}

int main() {
    
    double arr[5]={100.0, 500.0, 1234.0, 666.0, 2.0};
    solve();
    return 0;
}
#include <iostream>

void increment(int i, int j){
	while(i<=5){
		std::cout<<i++<<std::endl;	
	}
	std::cout<<"---------------------"<<std::endl;
	while(j<=5){
		std::cout<<++j<<std::endl;
	}
}

int main(){
	increment(1, 1);	
}

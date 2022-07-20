#include<iostream>
#include<array>


int* fillarr(int arr[]){
	return arr;
}

std::array<int, 5> func(){
	std::array<int, 5> n_array;
	for(int i=1; i<=5; i++){
		n_array[i] = i;
	}
	return n_array;
}

int main(){
	std::array<int, 5> arr;
	arr = func();
	std::cout<<"The array is ";
	for(int i=0; i<=5; i++){
		std::cout<<arr[i]<<std::endl;
	}	
	
	return 0;
}

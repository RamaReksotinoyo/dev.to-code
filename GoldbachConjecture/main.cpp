#include <iostream>

int primes[100000] = {2};
int j = 0;

int is_prime(int n);
void goldbach(int g);

int main(){
    int number = 0;
    while(1){
		std::cout<<"Enter even number:";
		std::cin>>number;
        if(number > 2 && number % 2==0){
            goldbach(number);
       		break;
	   	}
        else{
			std::cout<<"Incorrect number!"<<std::endl;
        }
        cout<<endl;
    }
    return 0;
}

int is_prime(int n){
    int flag = 1;
    for (int j = 2; j < n/2; j++){
        if((n%j) == 0){
            return flag-1;
        }
    }
    return flag;
}

void goldbach(int g){
    int flag = 0;

    if(primes[j]<g){
        for (int i = primes[j]+1; i < g; i++){
            if(is_prime(i) == 1){
                j++;
                primes[j] = i;
            }
        }       
    }

    for (int i = 0; i < j; i++){
        for (int k = 0; k < j; k++){
            if(primes[i] + primes[k] == g){
				std::cout<<g<<" = "<<primes[i]<<" + "<<primes[k]<<std::endl;
                break;
            }
        }
    }

}
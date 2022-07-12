#include <iostream>


int is_prime(int n){
    int num = 1;
    for (int i = 2; i < n/2; i++){
        if((n%i) == 0){
            return num-1;
        }
    }
    return num;
}

void solve(){
    int number = 0;
    int primes[100000] = {2};
    int j = 0;
    while(true){
		std::cout<<"Input even number:";
		std::cin>>number;
        if(number > 2 && number % 2==0){

            if(primes[j]<number){
                for (int i = primes[j]+1; i < number; i++){
                    if(is_prime(i) == 1){
                        j++;
                        primes[j] = i;
                    }
                }       
            }
            for (int i = 0; i < j; i++){
                for (int k = 0; k < j; k++){
                    if(primes[i] + primes[k] == number){
                        std::cout<<primes[i]<<" + "<<primes[k]<<" = "<<number<<std::endl;
                        break;
                    }
                }
            }
	   	}
        else{
			std::cout<<"Input must a even numbers"<<std::endl;
        }
    }
}

int main(){
    solve();
    return 0;
}



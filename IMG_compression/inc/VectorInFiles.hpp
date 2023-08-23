#include <fstream>
#include <vector>
#include <iostream>


class VectorInFiles {
public:
    template <class vecT> static void writeVector(std::vector<vecT>& vec, std::ofstream& writeFilePointer)
    {
        if(!writeFilePointer){
            std::cout << "Compression can't write on file. Opening file for writing error" << std::endl;
            throw 1;
        }

        // write vector size before the values
        unsigned int vecSize = vec.size();
        writeFilePointer.write(reinterpret_cast<char*>( &vecSize ), sizeof(unsigned int));

        
        // write vector values
        // without execution policy it executes in order
        // std::for_each(
        //     vec.begin(),
        //     vec.end(),
        //     [&](vecT &item) { 
        //         // std::cout << "item: " << item << std::endl;
        //         writeFilePointer.write(reinterpret_cast<char*>( &item ), sizeof(vecT)); 
        //     }
        // );
        writeFilePointer.write(reinterpret_cast<char*>( vec.data() ), sizeof(vecT)*vecSize); 

        // for(int i = 0; i < size1; i++){
        //     for(int j = 0; j < size2; j++){
        //         cdbl = m[i][j];
        //         std::cout << cdbl << ", ";
        //         oF.write( reinterpret_cast<char*>( &cdbl ), sizeof cdbl );
        //     }
        //     std::cout << std::endl;
        // }
    };

    template <class vecT> static void readVector(std::vector<vecT>& vec, std::ifstream& readFilePointer){
        if(!readFilePointer){
            std::cout << "Can't read file. Opening file for reading error" << std::endl;
            throw 1;
        }

        // raed vector size before the values
        unsigned int vecSize = 0;
        char vecSizeChar[sizeof(unsigned int)];

        if(!readFilePointer.read(vecSizeChar, sizeof(unsigned int))){
            std::cout << "Elements of the array are missing, file ends after vector size value." << std::endl;
            throw 2;
        }

        vecSize = *(reinterpret_cast<unsigned int *>(vecSizeChar));

        // for (unsigned int i = 0; i < sizeof(unsigned int); i++){
        //     vecSize |= (unsigned int) vecSizeChar[i] << (8*i);
        // }
        std::cout << "Vec size: " << vecSize << std::endl;

        vec.resize(vecSize,0);

        char dataArray[sizeof(vecT)*vecSize];

        readFilePointer.read(dataArray, sizeof(vecT)*vecSize);
        // for (unsigned int i = 0; i < sizeof(vecT)*vecSize; i++)
        //     dataArray[i] = (i+1)%sizeof(vecT)==0 ? 65: 0;
        

        vec.assign(reinterpret_cast<vecT *>( &dataArray[0]), reinterpret_cast<vecT *>( &dataArray[sizeof(vecT)*vecSize]) );

        // for (unsigned int i = 0; i < vecSize; i++){
        //     char itemChar[sizeof(vecT)];
        //     readFilePointer.read(itemChar, sizeof(vecT));
        //     if(!readFilePointer.getline(itemChar, sizeof(vecT))){
        //         std::cout << "Elements of the array are missing, file end reached" << std::endl;
        //         throw 2;
        //     }
                
        //     vecT temp = 0;
        //     for (unsigned int j = 0; j < sizeof(vecT); j++){
        //         temp |= (vecT) itemChar[j] << (8*j);
                
        //         // temp = temp | ( ((vecT) dataArray[i * sizeof(vecT) + j]) << (8*j) );
        //     }
        //     vec[i] = temp;
        // }
        

        // read vector values
        // std::for_each(
        //     vec.begin(),
        //     vec.end(),
        //     [](const vecT &item) { 
        //         writeFilePointer.write(reinterpret_cast<char*>( &item ), sizeof(vecT)); 
        //     }
        // );
        // writeFilePointer.write(reinterpret_cast<char*>( &vec.data() ), sizeof(vecT)*vecSize); 

    };
};
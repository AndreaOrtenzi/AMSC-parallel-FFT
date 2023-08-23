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
        writeFilePointer.write(reinterpret_cast<char*>( vec.data() ), sizeof(vecT)*vecSize);
    };

    template <class vecT> static void readVector(std::vector<vecT>& vec, std::ifstream& readFilePointer){
        if(!readFilePointer){
            std::cout << "Can't read file. Opening file for reading error" << std::endl;
            throw 1;
        }

        // read vector size before the values
        unsigned int vecSize = 0;
        char vecSizeChar[sizeof(unsigned int)];

        if(!readFilePointer.read(vecSizeChar, sizeof(unsigned int))){
            std::cout << "Elements of the array are missing, file ends after vector size value." << std::endl;
            throw 2;
        }

        vecSize = *(reinterpret_cast<unsigned int *>(vecSizeChar));
        vec.resize(vecSize,0);

        char dataArray[sizeof(vecT)*vecSize];

        readFilePointer.read(dataArray, sizeof(vecT)*vecSize);

        vec.assign(reinterpret_cast<vecT *>( &dataArray[0]), reinterpret_cast<vecT *>( &dataArray[sizeof(vecT)*vecSize]) );
    };
};
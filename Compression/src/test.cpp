/* File Handling with C++ using ifstream & ofstream class object*/
/* To write the Content in File*/
/* Then to read the content of file*/
#include <iostream>
#include <vector>
#include <algorithm>

#include "../inc/Compression.hpp"


using namespace std;

// Driver Code
int main()
{
    using T = double;

    vector<T> vec(5,64); // size, const val
    vector<T> vec2(5,1); // size, const val


    vector<unsigned char> encoded;
    vector<T> vals;
    vector<unsigned char> codes;
    vector<unsigned int> codesLen;

    {
        Compression<T> sam;

        T v = 2, v2 = 2;

        sam.add(2.48);
        sam.add(v);
        sam.add(vec[1]);
        sam.add(vec[1]);
        sam.add(v2);
        sam.add(v2); 
        sam.add(vec[1]);

        sam.getCompressed(encoded, vals, codes, codesLen);
        
        vector<T> hcVals(5,64);
        std::vector<unsigned int> hcTimes;
        
        sam.getHCData(hcVals, hcTimes);

        std::vector<T> rlVals;
        std::vector<unsigned int> rlTimes;
        sam.getRLvals(rlVals, rlTimes);

        std::cout << "RLVals:" << std::endl;
        std::for_each(
            rlVals.begin(),
            rlVals.end(),
            [](auto &item) { 
                cout << item << " "; 
            }
        );
        cout << endl;

        cout << "RLTimes:" << endl;
        std::for_each(
            rlTimes.begin(),
            rlTimes.end(),
            [](auto &item) { 
                cout << item << " "; 
            }
        );
        cout << endl;

        cout << "hcVals:" << endl;
        std::for_each(
            hcVals.begin(),
            hcVals.end(),
            [](auto &item) { 
                cout << item << " "; 
            }
        );
        cout << endl;

        cout << "hc freq:" << endl;
        std::for_each(
            hcTimes.begin(),
            hcTimes.end(),
            [](auto &item) { 
                cout << item << " "; 
            }
        );
        cout << endl;

        cout << "Encoded: " << encoded.size() << endl;
        std::for_each(
            encoded.begin(),
            encoded.end(),
            [](auto &item) { 
                cout << static_cast<unsigned int>(item) << " "; 
            }
        );
        cout << endl;

        cout << "Vals: " << vals.size() << endl;
        std::for_each(
            vals.begin(),
            vals.end(),
            [](auto &item) { 
                cout << item << " "; 
            }
        );
        cout << endl;

        cout << "codes: " << codes.size() << std::endl;
        std::for_each(
            codes.begin(),
            codes.end(),
            [](auto &item) { 
                cout << static_cast<unsigned int>(item) << " "; 
            }
        );
        cout << endl;

        cout << "Codes lenght: " << codesLen.size() << std::endl;
        std::for_each(
            codesLen.begin(),
            codesLen.end(),
            [](auto &item) { 
                cout << static_cast<unsigned int>(item) << " "; 
            }
        );
        cout << endl;
    }
	
    {
        std::vector<T> values;
        Compression<T> sam(encoded,vals,codes,codesLen);
        
        for (auto i = sam.begin(); i != sam.end(); i++)
            std::cout << *i << std::endl;
        
        std::cout << "Reverse " << std::endl;
        auto i = sam.end();
        while ( i != sam.begin()) {
            i--;
            std::cout << *i << std::endl;
        }
    }
    // by default ios::out mode, automatically deletes
	// the content of file. To append the content, open in ios:app
	// fout.open("sample.txt", ios::app)
	// fout.open("sample.txt");

    

    // VectorInFiles::writeVector<int>(vec,fout);

	// // Close the File
	// fout.close();

    // cout << "Prima:" << endl;
    // std::for_each(
    //     vec2.begin(),
    //     vec2.end(),
    //     [](const int &item) { 
    //         cout << item << " "; 
    //     }
    // );
    // cout << endl;

	// // Creation of ifstream class object to read the file
	// ifstream fin;

	// // by default open mode = ios::in mode
	// fin.open("sample.txt");

    // VectorInFiles::readVector<int>(vec2,fin);

    // cout << "Valori letti, valore attuale:" << endl;

    // std::for_each(
    //     vec2.begin(),
    //     vec2.end(),
    //     [](auto &item) { 
    //         cout << item << " "; 
    //     }
    // );
    // cout << endl;

	// // Close the file
	// fin.close();

	return 0;
}

#include <vector>
#include <complex>
#include <algorithm>
#include <fstream>
#include <iostream>

#include <set>
#include <unordered_map>

#ifndef COMPRESSED_TYPE
#define COMPRESSED_TYPE unsigned char // unsigned short MAX: 65535, unsigned char MAX: 255
#endif

// T must be more than unsigned int. I have to do T variable = unsigned int var;
template <class T> class Compression {
public:
    Compression(){};

    Compression(const std::vector<T> &vals){
        add(vals);
    };

    Compression(const std::vector<unsigned char> &encoded, const std::vector<T> &vals, const std::vector<COMPRESSED_TYPE> &codes, const std::vector<unsigned int> &codesLen){
        isHcComputed = false;

        rlValues.clear();
        rlTimes.clear();
        hcValues.clear();
        hcData.clear();
        
        COMPRESSED_TYPE code = 0;
        unsigned int codeLen = 0;
        bool isVal = true;
        T lastVal;

        for (unsigned int idx = 0; idx < encoded.size() - 2; ++idx){

            for (int j = 7; j >= 0; --j){
                code = (code<<1) | ((encoded[idx] & (1U << j))>>j);
                codeLen++;

                for (unsigned int w = 0; w < codes.size(); ++w){
                    if (code == codes[w] && codeLen == codesLen[w]) {
                        if (isVal)
                            lastVal = vals[w];
                        else {
                            for (unsigned int a = 0; a< vals[w]; ++a)
                                add(lastVal);
                        }
                        isVal = ! isVal;
                        code = 0;
                        codeLen = 0;
                    }
                }

            }
            
        }
        // last element gets only a part
        unsigned int idx = encoded.size() - 2;
        for (int j = 7; j >= static_cast<int>(8-encoded.back()); --j){
            code = (code<<1) | ((encoded[idx] & (1U << j))>>j);
            codeLen++;

            for (unsigned int w = 0; w < codes.size(); ++w){
                if (code == codes[w] && codeLen == codesLen[w]) {
                    if (isVal)
                            lastVal = vals[w];
                        else {
                            for (unsigned int a = 0; a< vals[w]; ++a)
                                add(lastVal);
                        }
                        isVal = ! isVal;
                        code = 0;
                        codeLen = 0;
                }
            }

        }
    };

    void add(const T& val);
    void add(const  std::vector<T>& vals); // a.reserve(a.size() + b.size() + c.size()); a.insert(a.end(), b.begin(), b.end());

    void getCompressed(std::vector<unsigned char> &encoded, std::vector<T>& vals, std::vector<COMPRESSED_TYPE> &codes, std::vector<unsigned int> &codesLen) const;
    void getCompressed(std::vector<unsigned char> &encoded, std::vector<T>& vals, std::vector<COMPRESSED_TYPE> &codes, std::vector<unsigned int> &codesLen);

    void getValues(std::vector<T>& vals) const {
        vals.clear();
        for (unsigned int i = 0; i< rlValues.size();++i){
            for (unsigned int j = 0; j<rlTimes[i]; ++j)
                vals.push_back(rlValues[i]);
        }
    }
    
    class HCInfo {
    public:
        HCInfo(): value(0)
            , frequency(1U)
            , code(0)
            , codeLen(0U)
            , sx(NULL)
            , dx(NULL)
            , isLeaf(true){
            std::cout << "PROBLEM: Called HCInfo without parameters, probably beacuse of :unordered_map::operator[]" << std::endl;
            throw 15;
        };
        HCInfo(T val)
            : value(val)
            , frequency(1U)
            , code(0)
            , codeLen(0U)
            , sx(NULL)
            , dx(NULL)
            , isLeaf(true){};

        HCInfo(HCInfo *i_sx, HCInfo *i_dx)
            : value(0)
            , frequency((*i_sx).frequency + (*i_dx).frequency)
            , code(0)
            , codeLen(0U)
            , sx(i_sx)
            , dx(i_dx)
            , isLeaf(false){};

        // recursive, expesive but easier to traverse the tree
        void assignCode() {
            if (!isLeaf){
                (*sx).assignCode(code,codeLen, false);
                (*dx).assignCode(code,codeLen, true);
            }            
        };

        void operator ++ (const int f) {
            frequency++;
        };
        void operator -- (const int f) {
            frequency--;
        };

        void operator += (const unsigned int f) {
            frequency += f;
        };

        bool operator > (const HCInfo& v) const {
            return frequency > v.frequency;
        };

        bool operator < (const HCInfo& v) const {
            return frequency < v.frequency;
        };

        bool operator == (const HCInfo &v) const {
            return value == v.value;
        };
        bool operator != (const HCInfo &v) const {
            return value != v.value;
        };

        bool operator == (const T &v) const {
            std::cout <<"Comparison between " << v << " and the value in the vector: " << value << std::endl;
            return value == v;
        };

        friend std::ostream& operator<<(std::ostream& os, const HCInfo& node) {
            std::cout << "Val: "<< node.value << ", Freq: " << node.frequency << 
                ", Code: "<< static_cast<unsigned int> (node.code) << ", Code length: "<< node.codeLen <<
                ", Leaf " << node.isLeaf << std::endl;
            return os;
        }

        ~HCInfo(){
            // std::cout << "Destroy node: " << *this;
            if(!isLeaf) {
                if(!(*dx).isLeaf)
                    delete dx;
                if(!(*sx).isLeaf)
                    delete sx;
            }
        }

        T value;
        // it doesn't affect ordering and comparisons in unordered sets
        // be carefull with ordered set
        unsigned int frequency; 
        COMPRESSED_TYPE code;
        unsigned int codeLen;

        // create a tree to assign codes
        HCInfo *sx, *dx;
        const bool isLeaf;
    private:
        void assignCode(const COMPRESSED_TYPE prevCode,const unsigned int prevCodeLen, const bool isRight) {

            codeLen = prevCodeLen + 1;
            code = (prevCode << 1) + isRight;

            if(!isLeaf) {
                (*sx).assignCode(code,codeLen, false);
                (*dx).assignCode(code,codeLen, true);
            }
        };

    };

    struct HCInfoPointerLessComparator {
        bool operator()(const HCInfo* a, const HCInfo* b) const {
            return *a < *b;
        }
    };

    // only for testing
    void traverseTree(const HCInfo &node, const unsigned int numDX, const unsigned int numSX);

    struct MyHash
    {
        std::size_t operator()(const HCInfo &h) const noexcept
        {
            return std::hash<int>()(h.value);
        }
    };

    // only for testing
    void getRLvals(std::vector<T> &vals,std::vector<unsigned int> &times){
        vals = rlValues;
        times = rlTimes;
    };

    // only for testing
    void getHCData(std::vector<T> &vals,std::vector<unsigned int> &times){
        vals.resize(hcData.size());
        times.resize(hcData.size());

        auto it = hcData.begin();

        for (unsigned int i = 0; i<hcData.size(); i++){
            vals[i] = (*it).second.value;
            times[i] = (*it).second.frequency;
            it++;
        }
    };

protected:
    void compressHC();
private:
    
    // Run-length encoding
    std::vector<T> rlValues;
    std::vector<unsigned int> rlTimes;

    // Huffman coding
    // Maximum value of COMPRESSED_TYPE indicates how many different values are allowed in the {rlValues,rlTimes}
    std::vector<unsigned char> hcValues; // secondo me qui non serve COMPRESSED_TYPE, meglio char sempre
    std::unordered_map<T,HCInfo, MyHash> hcData;

    bool isHcComputed = true;

    COMPRESSED_TYPE filter_array[8*sizeof(COMPRESSED_TYPE)];
};

#include "../src/Compression.tpp"
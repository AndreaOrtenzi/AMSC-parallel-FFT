
template <class T>
void Compression<T>::add(const T& i_val){
    T val = i_val >> approximation;
    // remove last value of rlTimes if it has frequency 1 because it's going to change => frequency 0
    if (isHcComputed && !rlTimes.empty()){
        T lastValueTimes = (T) rlTimes.back();
        if (hcData[lastValueTimes].frequency == 1)
            hcData.erase(lastValueTimes);
    }


    if (rlValues.empty() || rlValues.back() != val){
        
        // add HC
        if (hcData.count(val)){
            hcData[val]++; // when you modify it creates a new one and destroy the previous (not sure)
        } else {
            hcData.emplace(val, val); // even this creates and then destroyes an object, don't get why
        }

        if (!rlTimes.empty()) {
            T oldTimes = static_cast<T>(rlTimes.back());
            
            if ( hcData.count(oldTimes)){
                hcData[oldTimes]++;
            }else{ 
                hcData.emplace(oldTimes,oldTimes);
            }
        }

        // add RL
        rlValues.push_back(val);
        rlTimes.push_back(1);
    }else {
        rlTimes[rlTimes.size()-1]++;
    }

    isHcComputed = false;
}

template <class T>
void Compression<T>::add(const  std::vector<T>& vals){
    
    for (auto i = vals.begin(); i < vals.end(); i++)
        add(*i);
}

template <class T>
void Compression<T>::getCompressed(std::vector<unsigned char> &encoded, std::vector<T>& vals, std::vector<COMPRESSED_TYPE> &codes, std::vector<unsigned int> &codesLen) {
    if (!isHcComputed){
        compressHC();
    }
    
    ((const Compression<T>) *this).getCompressed(encoded,vals,codes, codesLen);
}

template <class T>
void Compression<T>::getCompressed(std::vector<unsigned char> &encoded, std::vector<T>& vals, std::vector<COMPRESSED_TYPE> &codes, std::vector<unsigned int> &codesLen) const {
    if (!isHcComputed)
        return;
    
    encoded = hcValues;
    
    unsigned int idx=0;
    vals.resize(hcData.size());
    codes.resize(hcData.size());
    codesLen.resize(hcData.size());
    
    for (auto i = hcData.begin(); i != hcData.end(); i++){
        vals[idx] = (*i).second.value;
        codes[idx] = (*i).second.code;
        codesLen[idx] = (*i).second.codeLen;
        idx++;
    }    
}


template <class T> 
void Compression<T>::traverseTree(const HCInfo &node, const unsigned int numDX, const unsigned int numSX){
    std::cout << "Node position: " << numDX << "dx " << numSX << "sx" << std::endl;
    std::cout << node;

    if (!node.isLeaf){
        //std::cout << "1SX: "<< node.sx << "1DX: " << node.dx << "1DX1SX: " << node.dx.sx << "2DX: " << node.dx.dx;
        traverseTree(*(node.sx),numDX,numSX+1);
        traverseTree(*(node.dx),numDX+1,numSX);
    }
}


template <class T>
void Compression<T>::compressHC(){
    T lastValueTimes = (T) rlTimes.back();
    if (hcData.count(lastValueTimes)){
        hcData[lastValueTimes]++; // when you modify it creates a new one and destroy the previous (not sure)
    } else {
        hcData.emplace(lastValueTimes, lastValueTimes); // even this creates and then destroyes an object, don't get why
    }

    // Assign codes to values
    // create the tree to from hcData based on frequency
    std::multiset<HCInfo*,HCInfoPointerLessComparator> hcDataTree;
    
    for (auto j = hcData.begin(); j != hcData.end(); j++){
        hcDataTree.insert(&((*j).second));
    }

    while(hcDataTree.size()>1){
        
        HCInfo *sxNode = *(hcDataTree.begin());
        hcDataTree.erase(hcDataTree.begin());

        HCInfo *dxNode = *(hcDataTree.begin());
        hcDataTree.erase(hcDataTree.begin());

        HCInfo *fatherNode = new HCInfo(sxNode,dxNode);
        hcDataTree.insert(fatherNode);

        // MANY copies here
    }
    
    HCInfo *treeHead = *(hcDataTree.begin());
    hcDataTree.erase(hcDataTree.begin());

    (*treeHead).assignCode();

    // traverseTree(*treeHead,0,0);
    
    // **********************
    // Use codes to write encoded array hcValues
    hcValues.resize(0);

    unsigned int lastCharIdx = 0;
    unsigned int availableBits = sizeof(COMPRESSED_TYPE)*8;
    constexpr unsigned int allocatedCOMPTYPE = 4;
    unsigned int allocatedAvailableBits = 0;

    for (unsigned int i = 0; i < rlValues.size(); i++) {
        const HCInfo &valInfo = hcData[rlValues[i]];
        COMPRESSED_TYPE codeValue = valInfo.code;
        unsigned int numBitsCodeV = valInfo.codeLen;

        const HCInfo &timeInfo = hcData[static_cast<T>(rlTimes[i])];
        COMPRESSED_TYPE codeTime = timeInfo.code;
        unsigned int numBitsCodeT = timeInfo.codeLen;

        if (allocatedAvailableBits < numBitsCodeV+numBitsCodeT){
            allocatedAvailableBits += allocatedCOMPTYPE*sizeof(COMPRESSED_TYPE)*8;
            hcValues.insert(hcValues.end(),allocatedCOMPTYPE*sizeof(COMPRESSED_TYPE),0);
        }
        
        COMPRESSED_TYPE *writePos;
        
        // encode rl value and add the code to the end of the hcValues
        writePos = reinterpret_cast<COMPRESSED_TYPE *>(&hcValues[lastCharIdx]);

        int spareBits = ((int) availableBits) - ((int) numBitsCodeV);
        if (spareBits >= 0){
            *writePos |= codeValue << spareBits;
            int usedChars = sizeof(COMPRESSED_TYPE) - (spareBits+7)/8;
            lastCharIdx += usedChars;
            availableBits = spareBits + usedChars*8;
        }else {
            spareBits = -spareBits;
            *writePos |= codeValue >> spareBits;

            lastCharIdx += sizeof(COMPRESSED_TYPE);
            writePos = reinterpret_cast<COMPRESSED_TYPE *>(&hcValues[lastCharIdx]);

            availableBits = sizeof(COMPRESSED_TYPE)*8-spareBits;
            *writePos |= codeValue << availableBits;
        }
        allocatedAvailableBits -= numBitsCodeV;

        // encode rl times and add the code to the end of the hcValues
        writePos = reinterpret_cast<COMPRESSED_TYPE *>(&hcValues[lastCharIdx]);

        spareBits = ((int) availableBits) - ((int) numBitsCodeT);
        if (spareBits >= 0){
            *writePos |= codeTime << spareBits;
            // std::cout << "Elem: " << rlTimes[i] << " i: " << i << " val in arr: " << static_cast<unsigned int>(*writePos) << std::endl;
            int usedChars = sizeof(COMPRESSED_TYPE) - (spareBits+7)/8;
            lastCharIdx += usedChars;
            availableBits = spareBits + usedChars*8;
        }else {
            spareBits = -spareBits;
            *writePos |= codeTime >> spareBits;

            lastCharIdx += sizeof(COMPRESSED_TYPE);
            writePos = reinterpret_cast<COMPRESSED_TYPE *>(&hcValues[lastCharIdx]);

            availableBits = sizeof(COMPRESSED_TYPE)*8-spareBits;
            *writePos |= codeTime << availableBits;
        }
        allocatedAvailableBits -= numBitsCodeT;
        
    }

    // use the last char to store the not used number of bits of the last char.
    unsigned char notUsedBits = availableBits%8;
    if (allocatedAvailableBits/8 == 0)
        hcValues.push_back(notUsedBits);
    else {
        lastCharIdx += (notUsedBits > 0);
        hcValues[lastCharIdx] = notUsedBits;
        hcValues.resize(lastCharIdx+1);
    }
    
    //hcFreqMap[lastValueTimes]--;
    delete treeHead;

    // restore frequency to allow other add() calls
    // if (hcData[lastValueTimes].frequency == 1)
    //     hcData.erase(lastValueTimes); // don't erase here but do it in add because I need it in the getCompressed array
    if (hcData[lastValueTimes].frequency > 1)
        hcData[lastValueTimes]--;

    isHcComputed = true;
}
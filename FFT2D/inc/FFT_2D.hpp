#include "AbstractFFT2D.hpp"

#include <complex>
#include <cmath>



class FFT_2D : public AbstractFFT2D
{
public: 
    
    FFT_2D(const Mat& sValues,const Mat& fValues) : AbstractFFT2D(sValues, fValues) {};

    const Mat& getSpatialValues() const override;

    const Mat& getFrequencyValues() const override;
  
    void transform() override;

    void iTransform() override;

private:

    void iterative_sequential(Mat& input_matrix, const unsigned int n);
    void recursive_sequential(Mat& input_matrix, const unsigned int n); 
    void iterative_parallel(Mat& input_matrix, const unsigned int n);
    void recursive_parallel(Mat& input_matrix, const unsigned int n);
    void recursive_seq_1D(SpVec& x, const unsigned n);
    void inv_transform_1D(SpVec& x, const unsigned n);
    static bool isRecursive;
    static bool isParallel;  

};

#include "AbstractFFT2D.hpp"

#include <complex>
#include <cmath>
#include <vector>


class FFT_2D : public AbstractFFT2D
{
public: 
    
    FFT_2D(const Mat& sValues,const Mat& fValues) : AbstractFFT2D(sValues, fValues) {};

    const Mat& getSpatialValues() const override;

    const Mat& getFrequencyValues() const override;
  
    void transform_par() override;

    void transform_seq() override;

    template <class C> void iterative_sequential(std::vector<std::vector<C>>& input_matrix, std::vector<std::vector<std::complex<double>>>& freq_matrix);

    void iTransform() override;

private:

    void iterative_sequential(Mat& input_matrix, const unsigned int n);

    void iterative_parallel(Mat& input_matrix, const unsigned int n);
    void inv_transform_1D(Vec& x);

};

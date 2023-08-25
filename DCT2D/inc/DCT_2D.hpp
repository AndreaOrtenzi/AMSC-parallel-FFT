#include "AbstractDCT2D.hpp"

#include <complex>
#include <cmath>



class DCT_2D : public AbstractDCT2D
{
public: 
    
    DCT_2D(const Mat& sValues,const Mat& fValues) : AbstractDCT2D(sValues, fValues) {};

    const Mat& getSpatialValues() const override;

    const Mat& getFrequencyValues() const override;
  
    void transform_par() override;

    void transform_seq() override;

    void iTransform() override;

private:

    void iterative_sequential(Mat& input_matrix, const unsigned int n); 
    void iterative_parallel(Mat& input_matrix, const unsigned int n);
    void inv_transform_1D(Vec& x);

};


template <class C> 
void SeqFFT2D::trasform(std::vector<std::vector<C>>& input_matrix, std::vector<std::vector<std::complex<double>>>& freq_matrix){

    if (input_matrix.empty())
        return;
        
    const unsigned int n_cols = input_matrix[0].size(), n_rows = input_matrix.size();

    unsigned int numBits = static_cast<unsigned int>(log2(n_cols));

    std::vector<std::complex<double>> col(n_rows,0.0);
    std::vector<std::vector<std::complex<double>>> input_cols(n_cols,col);
    freq_matrix.resize(n_rows);
    
    //First pass: Apply FFT to each row
    for (unsigned int i = 0; i < n_rows; ++i) {
        std::vector<std::complex<double>> &row_vector = freq_matrix[i];
        row_vector.resize(n_cols);

        
        for (unsigned int l = 0; l < n_cols; l++) { // **************
            unsigned int ji = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                std::swap(input_matrix[i][l], input_matrix[i][ji]);
            }
        }
        // use last iteration to write column vectors and the first to not override input_matrix
        // s = 1
        {
            unsigned int m = 1U << 1; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {

                // unsigned int ji = 0;
                // for (unsigned int l = 0; l < numBits; l++) {
                //     ji = (ji << 1) | ((k >> l) & 1U);
                // }
                // if (ji > k) {
                //     std::swap(input_matrix[i][k], input_matrix[i][ji]);
                // }

                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * static_cast<std::complex<double>>(input_matrix[i][k + j + m / 2]);
                    std::complex<double> u = static_cast<std::complex<double>>(input_matrix[i][k + j]);
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // swap again to restore original input_matrix
        
        for (unsigned int l = 0; l < n_cols; l++) { // **************
            unsigned int ji = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                std::swap(input_matrix[i][l], input_matrix[i][ji]);
            }
        }

        for (unsigned int s = 2; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // s == numBits
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    input_cols[k + j][i] = u + t;
                    input_cols[k + j + m / 2][i] = u - t;
                    w *= wm;
                }
            }
        }

        
        // input_matrix.row(i) = row_vector;
    }

    //Second pass: Apply FFT to each column
    numBits = static_cast<unsigned int>(log2(n_rows));
    for (unsigned int i = 0; i < n_cols; ++i) {
        std::vector<std::complex<double>> &col_vector = input_cols[i];
        
        for (unsigned int l = 0; l < n_rows; l++){
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(col_vector[l], col_vector[j]);
            }
        }

        for (unsigned int s = 1; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    col_vector[k + j] = u + t;
                    col_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // s == numBits
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    freq_matrix[k + j][i] = u + t;
                    freq_matrix[k + j + m / 2][i] = u - t;
                    w *= wm;
                }
            }
        }
    }
}
/*
template <class C> 
void iTransform1D(std::vector<std::complex<double>>& fValues, std::vector<C>& spatialValues) {
    // Perform the inverse Fourier transform on the frequency values and store the result in the spatial values
    // spatialValues.resize(N);
    // for (unsigned int n = 0; n < N; ++n) {
    //     std::complex<real> sum(0, 0);
    //     for (unsigned int k = 0; k < N; ++k) {
    //         std::complex<real> term = fValues[k] * std::exp(2.0 * M_PI * std::complex<real>(0, 1) * static_cast<real>(k * n) / static_cast<real>(N));
    //         sum += term;
    //     }
    //     spatialValues[n] = sum / static_cast<real>(N);
    // }
    spatialValues.resize(N);

    // Perform the inverse Fourier transform on the frequency values and store the result in the spatial values
    const unsigned int n = fValues.size();

    const unsigned int numBits = static_cast<unsigned int>(std::log2(n));
    for (unsigned int i = 0; i < n; i++) 
    {
        unsigned int j = 0;
        for (unsigned int k = 0; k < numBits; k++) {
            j = (j << 1) | ((i >> k) & 1U);
        }
        if (j > i) {
            std::swap(fValues[i], fValues[j]);
        }
    }
    for (unsigned int s = 1; s <= numBits; s++) {
        unsigned int m = 1U << s; 
        std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
        for (unsigned int k = 0; k < n; k += m) {
            std::complex<double> w = 1.0;
            for (unsigned int j = 0; j < m / 2; j++) {
                std::complex<double> t = w * fValues[k + j + m / 2];
                std::complex<double> u = fValues[k + j];
                spatialValues[k + j] = u + t;
                spatialValues[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }

    for (unsigned int i = 0; i < n; i++) 
    {
        spatialValues[i] /=  static_cast<double>(n);
        unsigned int j = 0;
        for (unsigned int k = 0; k < numBits; k++) {
            j = (j << 1) | ((i >> k) & 1U);
        }
        if (j > i) {
            std::swap(fValues[i], fValues[j]);
        }
    }
}
*/

template <class C> 
void SeqFFT2D::iTransform(std::vector<std::vector<std::complex<double>>>& input_matrix, std::vector<std::vector<C>>& space_matrix){
    if (input_matrix.empty())
        return;
        
    const unsigned int n_cols = input_matrix[0].size(), n_rows = input_matrix.size();

    unsigned int numBits = static_cast<unsigned int>(log2(n_cols));

    std::vector<std::complex<double>> col(n_rows,0.0);
    std::vector<std::vector<std::complex<double>>> input_cols(n_cols,col);
    space_matrix.resize(n_rows);
    
    //First pass: Apply iFFT to each row
    for (unsigned int i = 0; i < n_rows; ++i) {
        space_matrix[i].resize(n_cols);
        std::vector<std::complex<double>> row_vector = input_matrix[i];
        
        for (unsigned int l = 0; l < n_cols; l++) { // **************
            unsigned int ji = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                std::swap(row_vector[l], row_vector[ji]);
            }
        }

        for (unsigned int s = 1; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // s == numBits
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    input_cols[k + j][i] = (u + t) / static_cast<double>(n_cols);
                    input_cols[k + j + m / 2][i] = (u - t) /  static_cast<double>(n_cols);
                    w *= wm;
                }
            }
        }

        // input_matrix.row(i) = row_vector;
    }

    
    //Second pass: Apply FFT to each column
    numBits = static_cast<unsigned int>(log2(n_rows));
    for (unsigned int i = 0; i < n_cols; ++i) {
        std::vector<std::complex<double>> &col_vector = input_cols[i];
        
        for (unsigned int l = 0; l < n_rows; l++){
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(col_vector[l], col_vector[j]);
            }
        }

        for (unsigned int s = 1; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    col_vector[k + j] = u + t;
                    col_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        
        // s == numBits
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    // if (i == 1)
                    //     std::cout << (u + t).real()  / static_cast<double>(n_cols) << ", m/2: " << (u - t).real() / static_cast<double>(n_cols) << std::endl;
                    space_matrix[k + j][i] = static_cast<C>( (u + t).real() / static_cast<double>(n_rows) + 0.5);
                    space_matrix[k + j + m / 2][i] = static_cast<C>((u - t).real() / static_cast<double>(n_rows) + 0.5);
                    w *= wm;
                }
            }
        }
    }
}
/*
    // Perform the inverse Fourier transform on the frequency values and store the result in the spatial values
    const unsigned int n = input_matrix.size();
    space_matrix.resize(n);
    
    //First pass: apply inverse FFT1D on each row:
    for (unsigned int i = 0; i < n; ++i){
        std::vector<std::complex<double>> &row_vector = input_matrix[i];
        FFT_2D::inv_transform_1D(row_vector);
        spatialValues.row(i) = row_vector;
    }



    //Second pass: apply inverse FFT1D on each column:
    for (unsigned int i = 0; i < n; ++i){
        SpVec col_vector = frequencyValues.col(i);
        FFT_2D::inv_transform_1D(col_vector);
        spatialValues.col(i) = col_vector;
    }

    const auto t_f = high_resolution_clock::now();
    const auto time_inverse = duration_cast<microseconds>(t_f - t_i).count();
    std::cout << "*** Inverse Transform complete in " << time_inverse << " ms ***" << std::endl;
    std::cout << "---------------------------------------------------------------\n" << endl;

}*/